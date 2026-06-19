import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from drgrpo_grader import r1_zero_reward_fn
from train_sft import (
    build_optimizer,
    build_prompt,
    collate_sft_batch,
    dtype_from_name,
    load_sft_examples,
    save_checkpoint,
)
from train_rsft import batched, ground_truth_for_example, trim_after_stop


# 从题库中有放回随机抽取一个 rollout batch 的问题。
def sample_question_batch(examples, batch_size, rng):
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]


# 用当前 policy 作为 old policy，为每个问题采样 group_size 个回答。
def generate_grouped_responses(model, tokenizer, prompts, args, device):
    if args.group_size > 1 and args.sampling_temperature <= 0:
        raise ValueError("--group-size > 1 requires --sampling-temperature > 0.")

    was_training = model.training
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    grouped_responses = []
    for _, batch_prompts in tqdm(
        batched(prompts, args.generation_batch_size),
        total=(len(prompts) + args.generation_batch_size - 1)
        // args.generation_batch_size,
        desc="grpo generate",
    ):
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_model_len,
        ).to(device)

        generation_kwargs = {
            "max_new_tokens": args.sampling_max_tokens,
            "min_new_tokens": args.sampling_min_tokens,
            "num_return_sequences": args.group_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.sampling_temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.sampling_temperature,
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
            start = index * args.group_size
            end = start + args.group_size
            grouped_responses.append(decoded[start:end])

        del inputs, output_ids, generated_ids

    tokenizer.padding_side = original_padding_side
    if was_training:
        model.train()
    return grouped_responses


# 将 prompt 和某个采样 response 编码为 trajectory feature，并只在 response token 上训练。
def encode_rollout_feature(tokenizer, prompt, response, args):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    total_length = len(prompt_ids) + len(response_ids)
    if total_length > args.max_seq_length:
        if args.overflow_policy == "error":
            raise ValueError(
                "GRPO rollout is longer than max_seq_length. "
                f"total_length={total_length}, prompt_tokens={len(prompt_ids)}, "
                f"response_tokens={len(response_ids)}, max_seq_length={args.max_seq_length}. "
                "Increase --max-seq-length/max-model-len or reduce --sampling-max-tokens."
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
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "total_tokens": len(input_ids),
    }


# 按任务书公式 A_i = r_i - mean(r_1,...,r_G) 计算组内相对优势。
def compute_group_advantages(rewards, normalize, advantage_eps):
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    centered = reward_tensor - reward_tensor.mean()
    if normalize:
        centered = centered / (reward_tensor.std(unbiased=False) + advantage_eps)
    return [float(value) for value in centered]


# 对 rollout batch 编码并附加 reward/advantage 等训练元信息。
def build_rollout_features(tokenizer, question_batch, prompts, grouped_responses, args):
    rollout_features = []
    sample_records = []
    skipped = 0

    for question_offset, (example, prompt, responses) in enumerate(
        zip(question_batch, prompts, grouped_responses)
    ):
        ground_truth = ground_truth_for_example(example)
        rewards = []
        reward_infos = []
        for response in responses:
            reward_info = r1_zero_reward_fn(response, ground_truth, fast=args.fast_grade)
            reward_infos.append(reward_info)
            rewards.append(float(reward_info["reward"]))

        advantages = compute_group_advantages(
            rewards,
            normalize=args.normalize_advantage,
            advantage_eps=args.advantage_eps,
        )

        for generation_index, (response, reward_info, advantage) in enumerate(
            zip(responses, reward_infos, advantages)
        ):
            feature = encode_rollout_feature(tokenizer, prompt, response, args)
            accepted_feature = feature is not None
            if feature is None:
                skipped += 1
            else:
                feature.update(
                    {
                        "prompt": prompt,
                        "response": response,
                        "question": example["question"],
                        "final_answer": example["final_answer"],
                        "source_index": example["source_index"],
                        "question_offset": question_offset,
                        "generation_index": generation_index,
                        "reward": reward_info["reward"],
                        "format_reward": reward_info["format_reward"],
                        "answer_reward": reward_info["answer_reward"],
                        "advantage": advantage,
                    }
                )
                rollout_features.append(feature)

            sample_records.append(
                {
                    "source_index": example["source_index"],
                    "question_offset": question_offset,
                    "generation_index": generation_index,
                    "question": example["question"],
                    "ground_truth": ground_truth,
                    "final_answer": example["final_answer"],
                    "prompt": prompt,
                    "response": response,
                    "accepted_feature": accepted_feature,
                    "advantage": advantage,
                    **reward_info,
                }
            )

    return rollout_features, sample_records, skipped


# 计算每个 response token 在模型下的 log probability，不做序列求和，保留 token 级值。
def token_log_probs(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = batch["input_ids"][:, 1:]
    shift_mask = batch["response_mask"][:, 1:]
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    selected_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)
    return selected_log_probs, shift_mask, outputs


# 采样后立即用 old policy 计算固定的 token logprob，后续多 epoch 更新都以它为分母。
def attach_old_log_probs(model, tokenizer, features, args, device):
    model.eval()
    for start in range(0, len(features), args.train_batch_size):
        batch_features = features[start : start + args.train_batch_size]
        batch = collate_sft_batch(batch_features, tokenizer.pad_token_id, device)
        with torch.inference_mode():
            old_log_probs, shift_mask, _ = token_log_probs(model, batch)

        old_log_probs = old_log_probs.detach().cpu()
        shift_mask = shift_mask.detach().cpu()
        for row, feature in enumerate(batch_features):
            token_count = int(shift_mask[row].sum().item())
            if token_count <= 0:
                feature["old_log_probs"] = []
                continue
            feature["old_log_probs"] = old_log_probs[row][shift_mask[row].bool()].tolist()

        del batch, old_log_probs, shift_mask
    model.train()
    return [feature for feature in features if feature["old_log_probs"]]


# collate 时除 input_ids/mask 外，再补齐 old_log_probs 和 trajectory advantage。
def collate_grpo_batch(features, pad_token_id, device):
    batch = collate_sft_batch(features, pad_token_id, device)
    batch_size, seq_len = batch["input_ids"].shape
    old_log_probs = torch.zeros(
        (batch_size, seq_len - 1),
        dtype=torch.float32,
        device=device,
    )
    advantages = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for row, feature in enumerate(features):
        shift_response_mask = torch.tensor(feature["response_mask"][1:], dtype=torch.bool)
        positions = shift_response_mask.nonzero(as_tuple=False).squeeze(-1)
        values = torch.tensor(feature["old_log_probs"], dtype=torch.float32, device=device)
        if len(positions) != len(values):
            raise ValueError(
                "old_log_probs length does not match response token count: "
                f"{len(values)} vs {len(positions)}"
            )
        old_log_probs[row, positions.to(device=device, dtype=torch.long)] = values
        advantages[row] = float(feature["advantage"])

    batch["old_log_probs"] = old_log_probs
    batch["advantages"] = advantages
    return batch


# 按 GRPO-Clip 目标计算 loss：-mean(min(r*A, clip(r)*A))，A 对同一 trajectory 所有 token 相同。
def grpo_clip_loss(model, batch, clip_epsilon):
    current_log_probs, shift_mask, outputs = token_log_probs(model, batch)
    old_log_probs = batch["old_log_probs"]
    advantages = batch["advantages"].unsqueeze(1)

    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    token_objective = torch.minimum(unclipped_objective, clipped_objective)

    response_tokens = shift_mask.sum().clamp_min(1.0)
    trajectory_count = batch["advantages"].numel()
    loss = -(token_objective * shift_mask).sum() / max(trajectory_count, 1)

    with torch.no_grad():
        clipped = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float()
        approx_kl = ((old_log_probs - current_log_probs) * shift_mask).sum() / response_tokens
        metrics = {
            "loss": float(loss.detach().cpu()),
            "response_tokens": float(response_tokens.detach().cpu()),
            "trajectory_count": float(trajectory_count),
            "mean_ratio": float(((ratio * shift_mask).sum() / response_tokens).detach().cpu()),
            "clip_fraction": float(((clipped * shift_mask).sum() / response_tokens).detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
            "mean_advantage": float(batch["advantages"].mean().detach().cpu()),
            "mean_abs_advantage": float(batch["advantages"].abs().mean().detach().cpu()),
        }

    return loss, metrics, outputs


# 在固定 rollout batch 上执行若干次 GRPO 更新。
def train_on_rollout_batch(model, optimizer, tokenizer, rollout_features, args, device, rng):
    update_records = []
    if not rollout_features:
        return update_records

    for local_step in range(1, args.train_steps_per_rollout_batch + 1):
        step_loss = 0.0
        step_tokens = 0.0
        step_ratio = 0.0
        step_clip_fraction = 0.0
        step_kl = 0.0
        step_advantage = 0.0
        step_abs_advantage = 0.0

        model.train()
        for _ in range(args.gradient_accumulation_steps):
            micro_features = [
                rollout_features[rng.randrange(len(rollout_features))]
                for _ in range(args.train_batch_size)
            ]
            batch = collate_grpo_batch(micro_features, tokenizer.pad_token_id, device)
            loss, metrics, outputs = grpo_clip_loss(model, batch, args.clip_epsilon)
            (loss / args.gradient_accumulation_steps).backward()

            step_loss += metrics["loss"]
            step_tokens += metrics["response_tokens"]
            step_ratio += metrics["mean_ratio"]
            step_clip_fraction += metrics["clip_fraction"]
            step_kl += metrics["approx_kl"]
            step_advantage += metrics["mean_advantage"]
            step_abs_advantage += metrics["mean_abs_advantage"]

            del loss, outputs, batch

        grad_norm = None
        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            grad_norm = float(grad_norm.detach().cpu())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        scale = args.gradient_accumulation_steps
        update_records.append(
            {
                "local_train_step": local_step,
                "loss": step_loss / scale,
                "response_tokens": step_tokens / scale,
                "mean_ratio": step_ratio / scale,
                "clip_fraction": step_clip_fraction / scale,
                "approx_kl": step_kl / scale,
                "mean_advantage": step_advantage / scale,
                "mean_abs_advantage": step_abs_advantage / scale,
                "grad_norm": grad_norm,
            }
        )

    return update_records


# 解析命令行参数，默认值贴近任务书推荐，同时保留小规模 smoke test 所需开关。
def parse_args():
    parser = argparse.ArgumentParser(description="Handwritten GRPO training loop.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", choices=["math", "gsm8k"], default="math")
    parser.add_argument("--math-subject", default="algebra")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument(
        "--overflow-policy",
        choices=["error", "skip", "truncate"],
        default="error",
    )

    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--fast-grade", action="store_true")

    parser.add_argument("--normalize-advantage", action="store_true")
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--train-steps-per-rollout-batch", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
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
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-samples", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.group_size < 2:
        raise ValueError("GRPO requires --group-size >= 2 to compute relative advantages.")

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
    log_path = output_dir / "grpo_log.jsonl"
    samples_path = output_dir / "samples.jsonl"

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

    rng = random.Random(args.seed)
    start_time = time.time()
    total_generated = 0
    total_featured = 0
    global_train_step = 0

    with (
        log_path.open("w", encoding="utf-8") as log_file,
        samples_path.open("w", encoding="utf-8") as samples_file,
    ):
        for grpo_step in range(1, args.n_grpo_steps + 1):
            question_batch = sample_question_batch(
                raw_examples,
                args.rollout_batch_size,
                rng,
            )
            prompts = [build_prompt(example["question"]) for example in question_batch]

            grouped_responses = generate_grouped_responses(
                model,
                tokenizer,
                prompts,
                args,
                device,
            )
            rollout_features, sample_records, skipped = build_rollout_features(
                tokenizer,
                question_batch,
                prompts,
                grouped_responses,
                args,
            )

            total_generated += len(sample_records)
            total_featured += len(rollout_features)
            for record in sample_records:
                record["grpo_step"] = grpo_step
                samples_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            samples_file.flush()

            if rollout_features:
                rollout_features = attach_old_log_probs(
                    model,
                    tokenizer,
                    rollout_features,
                    args,
                    device,
                )

            reward_sum = sum(float(row["reward"]) for row in sample_records)
            format_sum = sum(float(row["format_reward"]) for row in sample_records)
            answer_sum = sum(float(row["answer_reward"]) for row in sample_records)
            advantage_values = [feature["advantage"] for feature in rollout_features]
            positive_advantages = sum(1 for value in advantage_values if value > 0)
            negative_advantages = sum(1 for value in advantage_values if value < 0)

            rollout_record = {
                "grpo_step": grpo_step,
                "event": "rollout",
                "questions": len(question_batch),
                "generated": len(sample_records),
                "features": len(rollout_features),
                "skipped": skipped,
                "format_acc": format_sum / max(len(sample_records), 1),
                "answer_acc": answer_sum / max(len(sample_records), 1),
                "mean_reward": reward_sum / max(len(sample_records), 1),
                "positive_advantages": positive_advantages,
                "negative_advantages": negative_advantages,
                "mean_abs_advantage": (
                    sum(abs(value) for value in advantage_values) / len(advantage_values)
                    if advantage_values
                    else 0.0
                ),
                "total_generated": total_generated,
                "total_features": total_featured,
                "elapsed_seconds": time.time() - start_time,
            }
            log_file.write(json.dumps(rollout_record, ensure_ascii=False) + "\n")
            log_file.flush()

            print(
                f"grpo_step={grpo_step} generated={rollout_record['generated']} "
                f"features={rollout_record['features']} "
                f"reward={rollout_record['mean_reward']:.4f} "
                f"format={rollout_record['format_acc']:.4f} "
                f"pos_adv={positive_advantages} neg_adv={negative_advantages}"
            )

            for record in sample_records[: args.print_samples if grpo_step == 1 else 0]:
                print("\n" + "=" * 80)
                print("question:", record["question"][:500])
                print("response:", record["response"][:1000])
                print(
                    "reward:",
                    {
                        "format_reward": record["format_reward"],
                        "answer_reward": record["answer_reward"],
                        "reward": record["reward"],
                    },
                )
                print("advantage:", record["advantage"])

            if not rollout_features:
                print(f"grpo_step={grpo_step} no usable rollout features; skip update")
                continue
            if positive_advantages == 0 and negative_advantages == 0:
                print(
                    f"grpo_step={grpo_step} all advantages are zero; "
                    "GRPO objective has no learning signal for this batch"
                )

            update_records = train_on_rollout_batch(
                model,
                optimizer,
                tokenizer,
                rollout_features,
                args,
                device,
                rng,
            )

            for update_record in update_records:
                global_train_step += 1
                update_record = {
                    "grpo_step": grpo_step,
                    "event": "grpo_update",
                    "global_train_step": global_train_step,
                    **update_record,
                    "elapsed_seconds": time.time() - start_time,
                }
                log_file.write(json.dumps(update_record, ensure_ascii=False) + "\n")
                log_file.flush()

                if global_train_step % args.log_every == 0:
                    print(
                        f"train_step={global_train_step} "
                        f"grpo_step={grpo_step} "
                        f"loss={update_record['loss']:.4f} "
                        f"ratio={update_record['mean_ratio']:.4f} "
                        f"clip={update_record['clip_fraction']:.4f} "
                        f"kl={update_record['approx_kl']:.4f} "
                        f"grad_norm={update_record['grad_norm'] if update_record['grad_norm'] is not None else 'NA'}"
                    )

            if args.save_every > 0 and grpo_step % args.save_every == 0:
                save_checkpoint(model, tokenizer, output_dir, f"step_{grpo_step}")

    if args.save_final:
        save_checkpoint(model, tokenizer, output_dir, "final")

    print("\nDone.")
    print(f"total_generated={total_generated}")
    print(f"total_features={total_featured}")
    print(f"logs: {log_path}")
    print(f"samples: {samples_path}")


if __name__ == "__main__":
    main()
