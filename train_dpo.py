import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_sft import (
    build_prompt,
    collate_sft_batch,
    dtype_from_name,
    save_checkpoint,
)


# 从 jsonl 文件中逐行读取记录，每一行是一个 JSON 对象。
def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# 加载偏好数据，支持两种格式：
# 1. 已经整理好的 chosen/rejected 偏好对；
# 2. RSFT/采样输出的 response/reward 记录，并自动按同题构造 winner-loser 对。
def load_preference_pairs(path, max_pairs):
    records = read_jsonl(path)
    if not records:
        raise RuntimeError(f"No records found in {path}")

    pairs = []
    if all("chosen" in item and "rejected" in item for item in records):
        for item in records:
            pairs.append(
                {
                    "question": item.get("question"),
                    "prompt": item.get("prompt"),
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "chosen_reward": item.get("chosen_reward"),
                    "rejected_reward": item.get("rejected_reward"),
                }
            )
    elif all("response" in item and "reward" in item for item in records):
        grouped = defaultdict(list)
        for item in records:
            key = item.get("source_index", item.get("question", item.get("prompt")))
            grouped[key].append(item)

        for group in grouped.values():
            winners = [item for item in group if item["reward"] > 0]
            losers = [item for item in group if item["reward"] <= 0]
            for winner in winners:
                for loser in losers:
                    pairs.append(
                        {
                            "question": winner.get("question"),
                            "prompt": winner.get("prompt"),
                            "chosen": winner["response"],
                            "rejected": loser["response"],
                            "chosen_reward": winner.get("reward"),
                            "rejected_reward": loser.get("reward"),
                        }
                    )
    else:
        raise ValueError(
            "Preference file must either contain direct records with "
            "`chosen` and `rejected`, or rollout records with `response` and `reward`."
        )

    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise RuntimeError(
            "No preference pairs were built. For rollout records, each prompt needs "
            "at least one reward>0 response and one reward<=0 response."
        )
    return pairs


# 把同一个 prompt 下的一条 response 编码成模型输入，并标出 response token 的 mask。
def encode_response(tokenizer, prompt, response, max_seq_length):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    total_length = len(prompt_ids) + len(response_ids)
    if total_length > max_seq_length:
        raise ValueError(
            "DPO example is longer than max_seq_length. "
            f"total_length={total_length}, prompt_tokens={len(prompt_ids)}, "
            f"response_tokens={len(response_ids)}, max_seq_length={max_seq_length}. "
            "Increase --max-seq-length or build shorter preference responses."
        )
    if not response_ids:
        raise ValueError("DPO response tokenization produced an empty response.")

    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "response_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "total_tokens": len(input_ids),
    }


# 将所有偏好对 tokenize 成 DPO 训练需要的 chosen/rejected feature。
def tokenize_preference_pairs(tokenizer, pairs, max_seq_length):
    features = []
    for pair in tqdm(pairs, desc="tokenize dpo"):
        prompt = pair["prompt"] if pair.get("prompt") else build_prompt(pair["question"])
        chosen = encode_response(tokenizer, prompt, pair["chosen"], max_seq_length)
        rejected = encode_response(tokenizer, prompt, pair["rejected"], max_seq_length)
        features.append(
            {
                "prompt": prompt,
                "question": pair.get("question"),
                "chosen_text": pair["chosen"],
                "rejected_text": pair["rejected"],
                "chosen": chosen,
                "rejected": rejected,
                "chosen_reward": pair.get("chosen_reward"),
                "rejected_reward": pair.get("rejected_reward"),
            }
        )

    max_total = max(
        max(item["chosen"]["total_tokens"], item["rejected"]["total_tokens"])
        for item in features
    )
    print(f"tokenized {len(features)} preference pairs; max_total_tokens={max_total}")
    return features


# 从偏好对 feature 列表中随机有放回采样一个 batch。
def sample_batch(features, batch_size, rng):
    return [features[rng.randrange(len(features))] for _ in range(batch_size)]


# 计算模型对完整 response 序列的 log probability：
# 只对 response_mask=1 的 token 位置求 log_softmax 并累加。
def sequence_log_probs(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = batch["input_ids"][:, 1:]
    shift_mask = batch["response_mask"][:, 1:]

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)
    response_log_probs = (token_log_probs * shift_mask).sum(dim=-1)
    response_tokens = shift_mask.sum(dim=-1).clamp_min(1.0)
    return response_log_probs, response_tokens


# 禁用 dropout，确保 policy/ref 的概率比较是确定性的。
def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


# 按任务书公式计算一个 batch 的 DPO loss 和诊断指标。
def dpo_loss(policy_model, ref_model, tokenizer, batch_features, beta, device):
    chosen_batch = collate_sft_batch(
        [item["chosen"] for item in batch_features],
        tokenizer.pad_token_id,
        device,
    )
    rejected_batch = collate_sft_batch(
        [item["rejected"] for item in batch_features],
        tokenizer.pad_token_id,
        device,
    )

    with torch.inference_mode():
        ref_chosen_logp, chosen_tokens = sequence_log_probs(ref_model, chosen_batch)
        ref_rejected_logp, rejected_tokens = sequence_log_probs(ref_model, rejected_batch)

    policy_chosen_logp, _ = sequence_log_probs(policy_model, chosen_batch)
    policy_rejected_logp, _ = sequence_log_probs(policy_model, rejected_batch)

    chosen_logratio = policy_chosen_logp - ref_chosen_logp
    rejected_logratio = policy_rejected_logp - ref_rejected_logp
    preference_logits = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(preference_logits).mean()

    with torch.no_grad():
        implicit_chosen_reward = beta * chosen_logratio
        implicit_rejected_reward = beta * rejected_logratio
        margin = preference_logits / beta
        metrics = {
            "loss": float(loss.detach().cpu()),
            "preference_accuracy": float((preference_logits > 0).float().mean().cpu()),
            "logit": float(preference_logits.mean().detach().cpu()),
            "logratio_margin": float(margin.mean().detach().cpu()),
            "policy_chosen_logp": float(policy_chosen_logp.mean().detach().cpu()),
            "policy_rejected_logp": float(policy_rejected_logp.mean().detach().cpu()),
            "ref_chosen_logp": float(ref_chosen_logp.mean().detach().cpu()),
            "ref_rejected_logp": float(ref_rejected_logp.mean().detach().cpu()),
            "implicit_chosen_reward": float(implicit_chosen_reward.mean().detach().cpu()),
            "implicit_rejected_reward": float(implicit_rejected_reward.mean().detach().cpu()),
            "chosen_tokens": float(chosen_tokens.float().mean().detach().cpu()),
            "rejected_tokens": float(rejected_tokens.float().mean().detach().cpu()),
        }

    return loss, metrics


# 定义命令行参数。只保留 DPO 训练本身需要的关键参数。
def parse_args():
    parser = argparse.ArgumentParser(description="Handwritten DPO training loop.")
    parser.add_argument("--model", required=True, help="Initial trainable policy model.")
    parser.add_argument(
        "--ref-model",
        default=None,
        help="Frozen reference model. Defaults to --model, which is the usual DPO start.",
    )
    parser.add_argument("--preference-file", required=True)
    parser.add_argument("--output-dir", default="checkpoints/dpo")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ref_model_name = args.ref_model or args.model

    # 固定随机种子，尽量保证采样 batch 和训练过程可复现。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 准备输出目录和训练日志文件。
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "dpo_log.jsonl"

    # 加载 tokenizer，并确保 padding 可用。DPO 会把不同长度样本 pad 成 batch。
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have either a pad token or an eos token.")

    # 读取偏好数据，并把 chosen/rejected response 编码成 token ids 与 response_mask。
    pairs = load_preference_pairs(args.preference_file, args.max_pairs)
    features = tokenize_preference_pairs(tokenizer, pairs, args.max_seq_length)

    # 分别加载可训练 policy 模型和冻结 reference 模型。
    dtype = dtype_from_name(args.dtype)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    policy_model.to(device)
    ref_model.to(device)
    policy_model.train()
    ref_model.eval()
    disable_dropout(policy_model)
    disable_dropout(ref_model)
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # 开启 gradient checkpointing 以降低训练显存占用；reference 不训练，只关闭 cache。
    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(policy_model.config, "use_cache"):
            policy_model.config.use_cache = False
        if hasattr(ref_model.config, "use_cache"):
            ref_model.config.use_cache = False

    # 构造优化器。这里只优化 policy_model，ref_model 参数已经冻结。
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    # 打印基础训练信息，方便确认 policy/ref、偏好对数量和参数规模。
    trainable_params = sum(
        param.numel() for param in policy_model.parameters() if param.requires_grad
    )
    total_params = sum(param.numel() for param in policy_model.parameters())
    print(f"device={device}; dtype={args.dtype}; beta={args.beta}")
    print(f"policy_model={args.model}")
    print(f"ref_model={ref_model_name}")
    print(f"preference_pairs={len(features)}")
    print(f"trainable_params={trainable_params:,}; total_params={total_params:,}")
    print(f"log_path={log_path}")

    # 主训练循环：随机取偏好对，计算 DPO loss，反向传播并更新 policy。
    rng = random.Random(args.seed)
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, args.train_steps + 1):
            step_loss = 0.0
            step_accuracy = 0.0
            step_logit = 0.0
            step_margin = 0.0

            for _ in range(args.gradient_accumulation_steps):
                micro_features = sample_batch(features, args.batch_size, rng)
                loss, metrics = dpo_loss(
                    policy_model,
                    ref_model,
                    tokenizer,
                    micro_features,
                    args.beta,
                    device,
                )
                (loss / args.gradient_accumulation_steps).backward()

                step_loss += metrics["loss"]
                step_accuracy += metrics["preference_accuracy"]
                step_logit += metrics["logit"]
                step_margin += metrics["logratio_margin"]

                del loss

            # 可选梯度裁剪，限制单步更新幅度，提升训练稳定性。
            grad_norm = None
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(),
                    args.max_grad_norm,
                )
                grad_norm = float(grad_norm.detach().cpu())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 写入训练日志，包括 DPO loss、偏好判断准确率、logit/margin 等诊断量。
            scale = args.gradient_accumulation_steps
            record = {
                "step": step,
                "loss": step_loss / scale,
                "preference_accuracy": step_accuracy / scale,
                "logit": step_logit / scale,
                "logratio_margin": step_margin / scale,
                "grad_norm": grad_norm,
                "elapsed_seconds": time.time() - start_time,
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()

            if step % args.log_every == 0:
                print(
                    f"step={step} loss={record['loss']:.4f} "
                    f"pref_acc={record['preference_accuracy']:.4f} "
                    f"logit={record['logit']:.4f} "
                    f"margin={record['logratio_margin']:.4f} "
                    f"grad_norm={grad_norm if grad_norm is not None else 'NA'}"
                )

    # 保存最终 policy checkpoint，后续评测时可直接作为 --model 路径。
    if args.save_final:
        save_checkpoint(policy_model, tokenizer, output_dir, "final")


if __name__ == "__main__":
    main()
