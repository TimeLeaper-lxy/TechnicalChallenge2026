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
            # 做一下转换 统一列名 这样是为了后面可以统一用一套逻辑
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

    # 如果参数传入是gsm8k而不是math
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

# 把列表切成小批次 按照batch size，yield一次返回一批
def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]

# transformer没有自动截断字符串到stop，需要手动截断（到</answer>）
def trim_after_stop(text: str, stop: str) -> str:
    stop_index = text.find(stop)
    if stop_index == -1:# 如果没找到 就原样返回
        return text
    return text[: stop_index + len(stop)]

# 把字符串 dtype 转成 torch dtype 
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

# 重点：生成函数
def generate_with_transformers(args, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # AutoTokenizer 用来加载 tokenizer。AutoModelForCausalLM 用来加载自回归语言模型。
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 为什么要用padding？因为同一个batch里面可能有多条prompt，每条长度可能不同，需要把他们组成一个矩阵。
    tokenizer.padding_side = "left"# 采用左padding的方式，即在左侧添加pad补齐长度
    # 如果tokenizer 没有 pad token，就用 eos token 代替（end ofsequence）
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    #加载模型
    dtype = dtype_from_name(args.dtype)#torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,#模型名
        dtype=dtype,#fp16
        attn_implementation="sdpa",#PyTorch 的 scaled dot product attention
    )
    #模型参数移动GPU eval是推理模式
    model.to(device)
    model.eval()

    all_responses = []
    for _, batch_prompts in tqdm(# tqdm给循环加进度条
        batched(prompts, args.batch_size),#调用之前定义的函数切分小批次 每次只有batchsize条prompt
        total=(len(prompts) + args.batch_size - 1) // args.batch_size,#计算batch数向上取整
        desc="generate",#进度条前面的描述文字
    ):
        # 把字符串变成模型输入
        inputs = tokenizer(# python语法，等价于tokenizer.__call__()
            batch_prompts,
            return_tensors="pt",#pytorch tensor
            padding=True,# 同一个batch内prompt补齐长度
            truncation=True,# 如果prompt太长就截断
            max_length=args.max_model_len,#最多保留这么多token输入 命令传参
        ).to(device)#返回的内容是一个字典

        with torch.inference_mode():# 表示不需要计算梯度（不需要训练，省显存）
            output_ids = model.generate(
                **inputs,#inputs字典展开成参数
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
    examples = load_examples(args)# 字典列表 前面提过
    if not examples:
        raise RuntimeError("No examples were loaded. Check dataset/split settings.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 可以看到 prompts就是经过两次填充两个常量提示词之后再组成列表  列表->列表可以用推导式
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
