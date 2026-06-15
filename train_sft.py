import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from drgrpo_grader import extract_answer


SYSTEM_PROMPT = """# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:
```{instruction}```

# Answer:
```"""

USER_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def build_prompt(question: str) -> str:
    instruction = USER_PROMPT.format(question=question)
    return SYSTEM_PROMPT.format(instruction=instruction)


def dtype_from_name(dtype_name: str):
    dtype_name = dtype_name.lower()
    if dtype_name in {"half", "float16", "fp16"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float", "float32", "fp32"}:
        return torch.float32
    if dtype_name == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def normalize_final_answer(answer: str) -> str:
    return answer.strip().replace("\n", " ")


def final_answer_from_solution(solution: str) -> str:
    if "\\boxed" in solution:
        boxed = extract_answer(solution)
        if boxed is not None:
            return normalize_final_answer(boxed)
    return normalize_final_answer(solution)


def build_sft_response(solution: str, final_answer: str) -> str:
    reasoning = solution.strip()
    final_answer = normalize_final_answer(final_answer)
    return f"\n{reasoning} </think> <answer>{final_answer}</answer>"


def select_example_range(examples, skip_samples, max_samples):
    if skip_samples > 0:
        examples = examples[skip_samples:]
    return examples[:max_samples] if max_samples else examples


def load_sft_examples(args, split=None, max_samples=None, skip_samples=0):
    split = args.split if split is None else split
    max_samples = args.max_samples if max_samples is None else max_samples
    if args.dataset == "math":
        subjects = (
            get_dataset_config_names("EleutherAI/hendrycks_math")
            if args.math_subject == "all"
            else [args.math_subject]
        )
        examples = []
        for subject in subjects:
            dataset = load_dataset(
                "EleutherAI/hendrycks_math",
                subject,
                split=split,
            )
            for row in dataset:
                final_answer = final_answer_from_solution(row["solution"])
                examples.append(
                    {
                        "dataset": "math",
                        "subject": subject,
                        "question": row["problem"],
                        "solution": row["solution"],
                        "final_answer": final_answer,
                    }
                )
        return select_example_range(examples, skip_samples, max_samples)

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    examples = []
    for row in dataset:
        parts = row["answer"].split("####")
        solution = parts[0].strip()
        final_answer = parts[-1].strip()
        examples.append(
            {
                "dataset": "gsm8k",
                "subject": "main",
                "question": row["question"],
                "solution": solution,
                "final_answer": final_answer,
            }
        )
    return select_example_range(examples, skip_samples, max_samples)


def encode_sft_example(tokenizer, example, max_seq_length, overflow_policy):
    prompt = build_prompt(example["question"])
    response = build_sft_response(example["solution"], example["final_answer"])

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    total_length = len(prompt_ids) + len(response_ids)
    if total_length > max_seq_length:
        if overflow_policy == "error":
            raise ValueError(
                "SFT example is longer than max_seq_length. "
                f"total_length={total_length}, prompt_tokens={len(prompt_ids)}, "
                f"response_tokens={len(response_ids)}, max_seq_length={max_seq_length}. "
                "Increase --max-seq-length, reduce the dataset, or explicitly set "
                "--overflow-policy skip/truncate if you intentionally accept that."
            )
        if overflow_policy == "skip":
            return None
        if overflow_policy == "truncate":
            response_room = max_seq_length - len(prompt_ids)
            if response_room <= 0:
                return None
            response_ids = response_ids[:response_room]
        else:
            raise ValueError(f"Unsupported overflow_policy: {overflow_policy}")

    input_ids = prompt_ids + response_ids
    response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

    if not response_ids:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "response_mask": response_mask,
        "prompt": prompt,
        "response": response,
        "question": example["question"],
        "final_answer": example["final_answer"],
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "total_tokens": len(input_ids),
    }


def summarize_token_lengths(tokenizer, examples):
    lengths = []
    for example in examples:
        prompt = build_prompt(example["question"])
        response = build_sft_response(example["solution"], example["final_answer"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]
        lengths.append((len(prompt_ids), len(response_ids), len(prompt_ids) + len(response_ids)))

    totals = sorted(total for _, _, total in lengths)
    if not totals:
        return {"count": 0}

    def percentile(percent):
        index = min(len(totals) - 1, int((len(totals) - 1) * percent))
        return totals[index]

    max_index = max(range(len(lengths)), key=lambda idx: lengths[idx][2])
    return {
        "count": len(lengths),
        "min_total": totals[0],
        "p50_total": percentile(0.50),
        "p90_total": percentile(0.90),
        "p95_total": percentile(0.95),
        "max_total": totals[-1],
        "max_prompt_tokens": lengths[max_index][0],
        "max_response_tokens": lengths[max_index][1],
        "max_example_index": max_index,
    }


def tokenize_sft_examples(tokenizer, examples, args, name="train"):
    length_summary = summarize_token_lengths(tokenizer, examples)
    print(f"{name} token length summary: {json.dumps(length_summary, ensure_ascii=False)}")
    if args.length_report_only:
        return []

    features = []
    skipped = 0
    for example in tqdm(examples, desc=f"tokenize {name}"):
        feature = encode_sft_example(
            tokenizer,
            example,
            max_seq_length=args.max_seq_length,
            overflow_policy=args.overflow_policy,
        )
        if feature is None:
            skipped += 1
            continue
        features.append(feature)

    if not features:
        raise RuntimeError("No tokenized SFT features were produced.")

    print(
        f"tokenized {name} "
        f"{len(features)} examples; skipped={skipped}; "
        f"max_total_tokens={max(item['total_tokens'] for item in features)}"
    )
    return features


def collate_sft_batch(features, pad_token_id, device):
    batch_size = len(features)
    max_len = max(len(item["input_ids"]) for item in features)

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )
    response_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.float32,
        device=device,
    )

    for row, item in enumerate(features):
        length = len(item["input_ids"])
        input_ids[row, :length] = torch.tensor(
            item["input_ids"], dtype=torch.long, device=device
        )
        attention_mask[row, :length] = torch.tensor(
            item["attention_mask"], dtype=torch.long, device=device
        )
        response_mask[row, :length] = torch.tensor(
            item["response_mask"], dtype=torch.float32, device=device
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


def sample_batch(features, batch_size, rng):
    return [features[rng.randrange(len(features))] for _ in range(batch_size)]


def masked_next_token_loss(logits, input_ids, response_mask):
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    label_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    response_tokens = shift_mask.sum().clamp_min(1.0)
    loss = -(label_log_probs * shift_mask).sum() / response_tokens

    with torch.no_grad():
        predictions = shift_logits.argmax(dim=-1)
        token_accuracy = ((predictions == shift_labels).float() * shift_mask).sum()
        token_accuracy = token_accuracy / response_tokens
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        response_entropy = (entropy * shift_mask).sum() / response_tokens

    return loss, {
        "response_tokens": float(response_tokens.detach().cpu()),
        "token_accuracy": float(token_accuracy.detach().cpu()),
        "response_entropy": float(response_entropy.detach().cpu()),
    }


def evaluate_features(model, features, tokenizer, device, batch_size):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0.0
    total_correct = 0.0
    total_entropy = 0.0

    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            micro_features = features[start : start + batch_size]
            batch = collate_sft_batch(micro_features, tokenizer.pad_token_id, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            loss, metrics = masked_next_token_loss(
                outputs.logits,
                batch["input_ids"],
                batch["response_mask"],
            )

            tokens = metrics["response_tokens"]
            total_loss += float(loss.detach().cpu()) * tokens
            total_tokens += tokens
            total_correct += metrics["token_accuracy"] * tokens
            total_entropy += metrics["response_entropy"] * tokens

            del outputs, loss, batch

    if was_training:
        model.train()

    total_tokens = max(total_tokens, 1.0)
    return {
        "eval_loss": total_loss / total_tokens,
        "eval_token_accuracy": total_correct / total_tokens,
        "eval_response_entropy": total_entropy / total_tokens,
        "eval_response_tokens": total_tokens,
        "eval_examples": len(features),
    }


def build_optimizer(model, args):
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.learning_rate)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def save_checkpoint(model, tokenizer, output_dir, step):
    checkpoint_dir = output_dir / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"saved checkpoint: {checkpoint_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Handwritten full-parameter SFT loop.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", choices=["math", "gsm8k"], default="math")
    parser.add_argument("--math-subject", default="algebra")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--skip-samples",
        type=int,
        default=0,
        help="Skip the first N examples before applying --max-samples.",
    )
    parser.add_argument("--output-dir", default="checkpoints/sft")
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument(
        "--overflow-policy",
        choices=["error", "skip", "truncate"],
        default="error",
        help="What to do when prompt+response exceeds max_seq_length.",
    )
    parser.add_argument(
        "--length-report-only",
        action="store_true",
        help="Only print token length statistics and exit before model loading.",
    )
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate a fixed tokenized set every N optimizer steps.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help=(
            "Dataset split for held-out loss evaluation, such as test. "
            "If omitted, eval uses the training features and is only an overfit diagnostic."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Batch size used for fixed loss evaluation.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="Evaluate only the first N examples; 0 means all available eval examples.",
    )
    parser.add_argument(
        "--eval-skip-samples",
        type=int,
        default=0,
        help="Skip the first N eval examples before applying --eval-max-samples.",
    )
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

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
    features = tokenize_sft_examples(tokenizer, raw_examples, args, name="train")
    if args.length_report_only:
        return

    eval_features = None
    eval_event_name = None
    if args.eval_every > 0:
        if args.eval_split:
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
            eval_event_name = "heldout_eval"
            print(
                f"heldout_eval uses split={args.eval_split}; "
                f"examples={len(eval_features)}"
            )
        else:
            eval_features = (
                features[: args.eval_max_samples]
                if args.eval_max_samples > 0
                else features
            )
            eval_event_name = "train_eval"
            print(
                "train_eval uses the same examples as training; "
                "this is an overfit diagnostic, not a generalization test."
            )

    dtype = dtype_from_name(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.to(device)
    model.train()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    optimizer = build_optimizer(model, args)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"device={device}; dtype={args.dtype}; optimizer={args.optimizer}")
    print(f"trainable_params={trainable_params:,}; total_params={total_params:,}")
    print(f"log_path={log_path}")

    rng = random.Random(args.seed)
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    with log_path.open("w", encoding="utf-8") as log_file:
        if args.eval_every > 0 and eval_features is not None:
            eval_metrics = evaluate_features(
                model,
                eval_features,
                tokenizer,
                device,
                args.eval_batch_size,
            )
            eval_record = {
                "step": 0,
                "event": eval_event_name,
                **eval_metrics,
                "elapsed_seconds": time.time() - start_time,
            }
            log_file.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
            log_file.flush()
            print(
                f"{eval_event_name} step=0 "
                f"loss={eval_metrics['eval_loss']:.4f} "
                f"tok_acc={eval_metrics['eval_token_accuracy']:.4f} "
                f"entropy={eval_metrics['eval_response_entropy']:.4f}"
            )

        for step in range(1, args.train_steps + 1):
            step_loss = 0.0
            step_tokens = 0.0
            step_accuracy = 0.0
            step_entropy = 0.0

            for _ in range(args.gradient_accumulation_steps):
                micro_features = sample_batch(features, args.batch_size, rng)
                batch = collate_sft_batch(micro_features, tokenizer.pad_token_id, device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                loss, metrics = masked_next_token_loss(
                    outputs.logits,
                    batch["input_ids"],
                    batch["response_mask"],
                )
                (loss / args.gradient_accumulation_steps).backward()

                step_loss += float(loss.detach().cpu())
                step_tokens += metrics["response_tokens"]
                step_accuracy += metrics["token_accuracy"]
                step_entropy += metrics["response_entropy"]

                del outputs, loss, batch

            grad_norm = None
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )
                grad_norm = float(grad_norm.detach().cpu())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            avg_loss = step_loss / args.gradient_accumulation_steps
            avg_tokens = step_tokens / args.gradient_accumulation_steps
            avg_accuracy = step_accuracy / args.gradient_accumulation_steps
            avg_entropy = step_entropy / args.gradient_accumulation_steps
            elapsed = time.time() - start_time

            record = {
                "step": step,
                "loss": avg_loss,
                "response_tokens": avg_tokens,
                "token_accuracy": avg_accuracy,
                "response_entropy": avg_entropy,
                "grad_norm": grad_norm,
                "elapsed_seconds": elapsed,
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()

            if step % args.log_every == 0:
                print(
                    f"step={step} "
                    f"loss={avg_loss:.4f} "
                    f"tok_acc={avg_accuracy:.4f} "
                    f"entropy={avg_entropy:.4f} "
                    f"grad_norm={grad_norm if grad_norm is not None else 'NA'}"
                )

            if (
                args.eval_every > 0
                and eval_features is not None
                and step % args.eval_every == 0
            ):
                eval_metrics = evaluate_features(
                    model,
                    eval_features,
                    tokenizer,
                    device,
                    args.eval_batch_size,
                )
                eval_record = {
                    "step": step,
                    "event": eval_event_name,
                    **eval_metrics,
                    "elapsed_seconds": time.time() - start_time,
                }
                log_file.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                log_file.flush()
                print(
                    f"{eval_event_name} step={step} "
                    f"loss={eval_metrics['eval_loss']:.4f} "
                    f"tok_acc={eval_metrics['eval_token_accuracy']:.4f} "
                    f"entropy={eval_metrics['eval_response_entropy']:.4f}"
                )

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(model, tokenizer, output_dir, step)

    if args.save_final:
        save_checkpoint(model, tokenizer, output_dir, "final")


if __name__ == "__main__":
    main()
