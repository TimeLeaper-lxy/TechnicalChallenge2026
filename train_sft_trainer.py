import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from train_sft import (
    dtype_from_name,
    evaluate_features,
    load_sft_examples,
    tokenize_sft_examples,
)


class SFTDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class SFTDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        batch_size = len(features)
        max_len = max(len(item["input_ids"]) for item in features)

        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)

        for row, item in enumerate(features):
            length = len(item["input_ids"])
            ids = torch.tensor(item["input_ids"], dtype=torch.long)
            response_mask = torch.tensor(item["response_mask"], dtype=torch.bool)
            input_ids[row, :length] = ids
            attention_mask[row, :length] = torch.tensor(
                item["attention_mask"],
                dtype=torch.long,
            )
            labels[row, :length] = torch.where(
                response_mask,
                ids,
                torch.full_like(ids, -100),
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ResponseOnlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        label_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        response_tokens = valid_mask.float().sum().clamp_min(1.0)
        loss = -(label_log_probs * valid_mask.float()).sum() / response_tokens

        return (loss, outputs) if return_outputs else loss


class JSONLLogCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.start_time = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def _write(self, record):
        record["elapsed_seconds"] = time.time() - self.start_time
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step}
        for key, value in logs.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                continue
            record[key] = value
        self._write(record)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        record = {"step": state.global_step, "event": "heldout_eval"}
        for key, value in metrics.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                continue
            record[key] = value
        self._write(record)


class DetailedEvalCallback(TrainerCallback):
    def __init__(self, eval_features, tokenizer, batch_size, log_path):
        self.eval_features = eval_features
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.log_path = Path(log_path)
        self.start_time = time.time()
        self.last_eval_step = None

    def _write(self, record):
        record["elapsed_seconds"] = time.time() - self.start_time
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _evaluate(self, model, step):
        device = next(model.parameters()).device
        metrics = evaluate_features(
            model,
            self.eval_features,
            self.tokenizer,
            device,
            self.batch_size,
        )
        self._write(
            {
                "step": step,
                "event": "heldout_eval",
                **metrics,
            }
        )
        print(
            f"heldout_eval step={step} "
            f"loss={metrics['eval_loss']:.4f} "
            f"tok_acc={metrics['eval_token_accuracy']:.4f} "
            f"entropy={metrics['eval_response_entropy']:.4f}"
        )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.last_eval_step = 0
            self._evaluate(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or args.eval_steps is None or args.eval_steps <= 0:
            return
        step = state.global_step
        if step > 0 and step % args.eval_steps == 0 and step != self.last_eval_step:
            self.last_eval_step = step
            self._evaluate(model, step)


def parse_args():
    parser = argparse.ArgumentParser(description="Trainer-based response-only SFT baseline.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", choices=["math", "gsm8k"], default="math")
    parser.add_argument("--math-subject", default="algebra")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--output-dir", default="checkpoints/sft_trainer")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--overflow-policy",
        choices=["error", "skip", "truncate"],
        default="error",
    )
    parser.add_argument("--length-report-only", action="store_true")
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--eval-skip-samples", type=int, default=0)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have either a pad token or an eos token.")

    raw_examples = load_sft_examples(
        args,
        split=args.split,
        max_samples=args.max_samples,
        skip_samples=args.skip_samples,
    )
    train_features = tokenize_sft_examples(tokenizer, raw_examples, args, name="train")
    if args.length_report_only:
        return

    eval_features = None
    if args.eval_every > 0:
        if not args.eval_split:
            raise ValueError("--eval-split is required for Trainer comparison.")
        raw_eval_examples = load_sft_examples(
            args,
            split=args.eval_split,
            max_samples=args.eval_max_samples,
            skip_samples=args.eval_skip_samples,
        )
        eval_features = tokenize_sft_examples(
            tokenizer,
            raw_eval_examples,
            args,
            name=f"eval:{args.eval_split}",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_from_name(args.dtype),
        attn_implementation=args.attn_implementation,
    )

    if args.gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.train_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.dtype.lower() in {"bfloat16", "bf16"},
        fp16=args.dtype.lower() in {"float16", "fp16", "half"},
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.log_every,
        logging_strategy="steps",
        eval_strategy="no",
        eval_steps=args.eval_every if args.eval_every > 0 else None,
        per_device_eval_batch_size=args.eval_batch_size,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_drop_last=False,
    )

    callbacks = [JSONLLogCallback(output_dir / "train_log.jsonl")]
    if eval_features is not None:
        callbacks.append(
            DetailedEvalCallback(
                eval_features,
                tokenizer,
                args.eval_batch_size,
                output_dir / "train_log.jsonl",
            )
        )

    trainer = ResponseOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(train_features),
        eval_dataset=None,
        data_collator=SFTDataCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    trainer.train()

    if args.save_final:
        final_dir = output_dir / "step_final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)
        print(f"saved checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
