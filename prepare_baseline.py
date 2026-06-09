import argparse
import json
import os
import sys
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm

from drgrpo_grader import r1_zero_reward_fn


def configure_pip_cuda_home():
    """Let FlashInfer find the nvcc installed by pip's nvidia-cuda-nvcc package."""
    venv_bin = Path(sys.prefix) / "bin"
    if venv_bin.exists():
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return

    cuda_home = (
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages" / "nvidia" / "cu13"
    )
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.exists():
        return

    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["CUDA_PATH"] = str(cuda_home)
    os.environ["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"


configure_pip_cuda_home()


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


def load_examples(args):
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
                split=args.split,
            )
            for row in dataset:
                examples.append(
                    {
                        "dataset": "math",
                        "subject": subject,
                        "question": row["problem"],
                        "ground_truth": row["solution"],
                    }
                )
        return examples[: args.max_samples] if args.max_samples else examples

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    examples = []
    for row in dataset:
        answer = row["answer"].split("####")[-1].strip()
        examples.append(
            {
                "dataset": "gsm8k",
                "subject": "main",
                "question": row["question"],
                "ground_truth": answer,
            }
        )
    return examples[: args.max_samples] if args.max_samples else examples


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def trim_after_stop(text: str, stop: str) -> str:
    stop_index = text.find(stop)
    if stop_index == -1:
        return text
    return text[: stop_index + len(stop)]


def dtype_from_name(dtype_name: str):
    import torch

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


def generate_with_transformers(args, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = dtype_from_name(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    all_responses = []
    for _, batch_prompts in tqdm(
        batched(prompts, args.batch_size),
        total=(len(prompts) + args.batch_size - 1) // args.batch_size,
        desc="generate",
    ):
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_model_len,
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_responses.extend(trim_after_stop(text, "</answer>") for text in responses)

    return all_responses


def generate_with_vllm(args, prompts):
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )

    responses = []
    for _, batch_prompts in tqdm(
        batched(prompts, args.batch_size),
        total=(len(prompts) + args.batch_size - 1) // args.batch_size,
        desc="generate",
    ):
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        responses.extend(output.outputs[0].text for output in outputs)
    return responses


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the preparation-stage zero-shot baseline."
    )
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", choices=["math", "gsm8k"], default="math")
    parser.add_argument("--math-subject", default="algebra")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/baseline_math_algebra.jsonl")
    parser.add_argument("--fast-grade", action="store_true")
    parser.add_argument("--print-samples", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_examples(args)
    if not examples:
        raise RuntimeError("No examples were loaded. Check dataset/split settings.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = [build_prompt(item["question"]) for item in examples]
    if args.backend == "vllm":
        responses = generate_with_vllm(args, prompts)
    else:
        responses = generate_with_transformers(args, prompts)

    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    with output_path.open("w", encoding="utf-8") as f:
        for index, (item, prompt, response) in enumerate(zip(examples, prompts, responses)):
            reward = r1_zero_reward_fn(
                response,
                item["ground_truth"],
                fast=args.fast_grade,
            )
            record = {
                "index": index,
                **item,
                "prompt": prompt,
                "response": response,
                **reward,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_reward += reward["reward"]
            total_format_reward += reward["format_reward"]
            total_answer_reward += reward["answer_reward"]

            if index < args.print_samples:
                print("\n" + "=" * 80)
                print(f"sample #{index + 1}")
                print("question:", item["question"][:500])
                print("response:", response[:1000])
                print("reward:", reward)

    print("\nDone.")
    print(f"results: {output_path}")
    print(f"samples: {len(responses)}")
    print(f"format_acc: {total_format_reward / len(responses):.4f}")
    print(f"answer_acc: {total_answer_reward / len(responses):.4f}")
    print(f"reward: {total_reward / len(responses):.4f}")


if __name__ == "__main__":
    main()
