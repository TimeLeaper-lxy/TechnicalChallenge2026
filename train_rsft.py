import argparse
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from drgrpo_grader import r1_zero_reward_fn
from train_sft import (
    build_optimizer,
    build_prompt,
    collate_sft_batch,
    dtype_from_name,
    evaluate_features,
    load_sft_examples,
    masked_next_token_loss,
    save_checkpoint,
    tokenize_sft_examples,
)


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def trim_after_stop(text: str, stop: str) -> str:
    stop_index = text.find(stop)
    if stop_index == -1:
        return text
    return text[: stop_index + len(stop)]


def ground_truth_for_example(example):
    if example["dataset"] == "math":
        return example["solution"]
    return example["final_answer"]


def sample_question_batch(examples, batch_size, rng):
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]


def generate_candidate_responses(model, tokenizer, prompts, args, device):
    if args.num_generations > 1 and args.temperature <= 0:
        raise ValueError("--num-generations > 1 requires --temperature > 0.")

    was_training = model.training
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    grouped_responses = []
    for _, batch_prompts in tqdm(
        batched(prompts, args.generation_batch_size),
        total=(len(prompts) + args.generation_batch_size - 1)
        // args.generation_batch_size,
        desc="rsft generate",
    ):
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_model_len,
        ).to(device)

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "num_return_sequences": args.num_generations,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = [trim_after_stop(text, "</answer>") for text in decoded]

        for index in range(len(batch_prompts)):
            start = index * args.num_generations
            end = start + args.num_generations
            grouped_responses.append(decoded[start:end])

        del inputs, output_ids, generated_ids

    tokenizer.padding_side = original_padding_side
    if was_training:
        model.train()
    return grouped_responses


def encode_rsft_example(tokenizer, example, response, args):
    prompt = build_prompt(example["question"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    total_length = len(prompt_ids) + len(response_ids)
    if total_length > args.max_seq_length:
        if args.overflow_policy == "error":
            raise ValueError(
                "RSFT accepted example is longer than max_seq_length. "
                f"total_length={total_length}, prompt_tokens={len(prompt_ids)}, "
                f"response_tokens={len(response_ids)}, max_seq_length={args.max_seq_length}. "
                "Increase --max-seq-length, reduce --max-new-tokens, or explicitly set "
                "--overflow-policy skip/truncate if you intentionally accept that."
            )
        if args.overflow_policy == "skip":
            return None
        if args.overflow_policy == "truncate":
            response_room = args.max_seq_length - len(prompt_ids)
            if response_room <= 0:
                return None
            response_ids = response_ids[:response_room]
        else:
            raise ValueError(f"Unsupported overflow_policy: {args.overflow_policy}")

    if not response_ids:
        return None

    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "response_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
        "prompt": prompt,
        "response": response,
        "question": example["question"],
        "final_answer": example["final_answer"],
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "total_tokens": len(input_ids),
    }


def train_one_sft_step(model, optimizer, tokenizer, features, args, device, rng):
    step_loss = 0.0
    step_tokens = 0.0
    step_accuracy = 0.0
    step_entropy = 0.0

    model.train()
    for _ in range(args.gradient_accumulation_steps):
        micro_features = [
            features[rng.randrange(len(features))] for _ in range(args.batch_size)
        ]
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        grad_norm = float(grad_norm.detach().cpu())

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    scale = args.gradient_accumulation_steps
    return {
        "loss": step_loss / scale,
        "response_tokens": step_tokens / scale,
        "token_accuracy": step_accuracy / scale,
        "response_entropy": step_entropy / scale,
        "grad_norm": grad_norm,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Plain handwritten RSFT loop.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", choices=["math", "gsm8k"], default="math")
    parser.add_argument("--math-subject", default="algebra")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--output-dir", default="checkpoints/rsft")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--overflow-policy",
        choices=["error", "skip", "truncate"],
        default="error",
    )
    parser.add_argument(
        "--length-report-only",
        action="store_true",
        help="Only print token length statistics for eval data and exit.",
    )

    parser.add_argument("--rsft-rounds", type=int, default=1)
    parser.add_argument("--question-batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--reward-threshold", type=float, default=1.0)
    parser.add_argument("--fast-grade", action="store_true")

    parser.add_argument("--sft-steps-per-round", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
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

    parser.add_argument("--eval-every-rounds", type=int, default=0)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--eval-skip-samples", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=1)

    parser.add_argument("--save-every-rounds", type=int, default=0)
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--print-samples", type=int, default=2)
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
    log_path = output_dir / "rsft_log.jsonl"
    samples_path = output_dir / "samples.jsonl"
    accepted_path = output_dir / "accepted.jsonl"

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
    for index, example in enumerate(raw_examples):
        example["source_index"] = args.skip_samples + index
    if not raw_examples:
        raise RuntimeError("No training questions were loaded.")

    eval_features = None
    if args.eval_every_rounds > 0:
        if args.eval_split is None:
            raise ValueError("--eval-every-rounds requires --eval-split.")
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
        if args.length_report_only:
            return

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
    optimizer.zero_grad(set_to_none=True)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"device={device}; dtype={args.dtype}; optimizer={args.optimizer}")
    print(f"trainable_params={trainable_params:,}; total_params={total_params:,}")
    print(f"log_path={log_path}")
    print(f"samples_path={samples_path}")
    print(f"accepted_path={accepted_path}")

    rng = random.Random(args.seed)
    start_time = time.time()
    global_sft_step = 0
    total_generated = 0
    total_accepted = 0

    with (
        log_path.open("w", encoding="utf-8") as log_file,
        samples_path.open("w", encoding="utf-8") as samples_file,
        accepted_path.open("w", encoding="utf-8") as accepted_file,
    ):
        if args.eval_every_rounds > 0 and eval_features is not None:
            eval_metrics = evaluate_features(
                model,
                eval_features,
                tokenizer,
                device,
                args.eval_batch_size,
            )
            record = {
                "round": 0,
                "event": "heldout_eval",
                **eval_metrics,
                "elapsed_seconds": time.time() - start_time,
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            print(
                "heldout_eval round=0 "
                f"loss={eval_metrics['eval_loss']:.4f} "
                f"tok_acc={eval_metrics['eval_token_accuracy']:.4f} "
                f"entropy={eval_metrics['eval_response_entropy']:.4f}"
            )

        for round_index in range(1, args.rsft_rounds + 1):
            question_batch = sample_question_batch(
                raw_examples,
                args.question_batch_size,
                rng,
            )
            prompts = [build_prompt(example["question"]) for example in question_batch]
            grouped_responses = generate_candidate_responses(
                model,
                tokenizer,
                prompts,
                args,
                device,
            )

            round_features = []
            round_generated = 0
            round_accepted = 0
            reward_sum = 0.0
            format_sum = 0.0
            answer_sum = 0.0

            for question_offset, (example, prompt, responses) in enumerate(
                zip(question_batch, prompts, grouped_responses)
            ):
                ground_truth = ground_truth_for_example(example)
                for generation_index, response in enumerate(responses):
                    reward = r1_zero_reward_fn(
                        response,
                        ground_truth,
                        fast=args.fast_grade,
                    )
                    accepted = reward["reward"] >= args.reward_threshold
                    round_generated += 1
                    total_generated += 1
                    reward_sum += reward["reward"]
                    format_sum += reward["format_reward"]
                    answer_sum += reward["answer_reward"]

                    sample_record = {
                        "round": round_index,
                        "source_index": example["source_index"],
                        "question_offset": question_offset,
                        "generation_index": generation_index,
                        "question": example["question"],
                        "ground_truth": ground_truth,
                        "final_answer": example["final_answer"],
                        "prompt": prompt,
                        "response": response,
                        "accepted": accepted,
                        **reward,
                    }
                    samples_file.write(json.dumps(sample_record, ensure_ascii=False) + "\n")

                    if not accepted:
                        continue

                    feature = encode_rsft_example(tokenizer, example, response, args)
                    if feature is None:
                        continue
                    round_features.append(feature)
                    round_accepted += 1
                    total_accepted += 1
                    accepted_file.write(
                        json.dumps(sample_record, ensure_ascii=False) + "\n"
                    )

                    if total_accepted <= args.print_samples:
                        print("\n" + "=" * 80)
                        print(f"accepted sample #{total_accepted}")
                        print("question:", example["question"][:500])
                        print("response:", response[:1000])
                        print("reward:", reward)

            samples_file.flush()
            accepted_file.flush()

            round_record = {
                "round": round_index,
                "event": "sample_filter",
                "generated": round_generated,
                "accepted": round_accepted,
                "acceptance_rate": round_accepted / max(round_generated, 1),
                "format_acc": format_sum / max(round_generated, 1),
                "answer_acc": answer_sum / max(round_generated, 1),
                "mean_reward": reward_sum / max(round_generated, 1),
                "total_generated": total_generated,
                "total_accepted": total_accepted,
                "elapsed_seconds": time.time() - start_time,
            }
            log_file.write(json.dumps(round_record, ensure_ascii=False) + "\n")
            log_file.flush()
            print(
                f"round={round_index} generated={round_generated} "
                f"accepted={round_accepted} "
                f"accept_rate={round_record['acceptance_rate']:.4f} "
                f"format_acc={round_record['format_acc']:.4f} "
                f"answer_acc={round_record['answer_acc']:.4f}"
            )

            if not round_features:
                print(f"round={round_index} no accepted samples; skip SFT update")
            else:
                for local_step in range(1, args.sft_steps_per_round + 1):
                    global_sft_step += 1
                    metrics = train_one_sft_step(
                        model,
                        optimizer,
                        tokenizer,
                        round_features,
                        args,
                        device,
                        rng,
                    )
                    update_record = {
                        "round": round_index,
                        "event": "sft_update",
                        "round_sft_step": local_step,
                        "global_sft_step": global_sft_step,
                        "round_train_examples": len(round_features),
                        **metrics,
                        "elapsed_seconds": time.time() - start_time,
                    }
                    log_file.write(json.dumps(update_record, ensure_ascii=False) + "\n")
                    log_file.flush()

                    if global_sft_step % args.log_every == 0:
                        print(
                            f"rsft_sft_step={global_sft_step} "
                            f"round={round_index} "
                            f"loss={metrics['loss']:.4f} "
                            f"tok_acc={metrics['token_accuracy']:.4f} "
                            f"entropy={metrics['response_entropy']:.4f} "
                            f"grad_norm={metrics['grad_norm'] if metrics['grad_norm'] is not None else 'NA'}"
                        )

            if (
                args.eval_every_rounds > 0
                and eval_features is not None
                and round_index % args.eval_every_rounds == 0
            ):
                eval_metrics = evaluate_features(
                    model,
                    eval_features,
                    tokenizer,
                    device,
                    args.eval_batch_size,
                )
                eval_record = {
                    "round": round_index,
                    "event": "heldout_eval",
                    **eval_metrics,
                    "elapsed_seconds": time.time() - start_time,
                }
                log_file.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                log_file.flush()
                print(
                    f"heldout_eval round={round_index} "
                    f"loss={eval_metrics['eval_loss']:.4f} "
                    f"tok_acc={eval_metrics['eval_token_accuracy']:.4f} "
                    f"entropy={eval_metrics['eval_response_entropy']:.4f}"
                )

            if args.save_every_rounds > 0 and round_index % args.save_every_rounds == 0:
                save_checkpoint(model, tokenizer, output_dir, f"round_{round_index}")

    if args.save_final:
        save_checkpoint(model, tokenizer, output_dir, "final")

    print("\nDone.")
    print(f"total_generated={total_generated}")
    print(f"total_accepted={total_accepted}")
    print(f"acceptance_rate={total_accepted / max(total_generated, 1):.4f}")
    print(f"logs: {log_path}")
    print(f"samples: {samples_path}")
    print(f"accepted: {accepted_path}")


if __name__ == "__main__":
    main()
