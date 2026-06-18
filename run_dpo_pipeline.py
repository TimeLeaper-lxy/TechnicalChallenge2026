import argparse
import json
import subprocess
import time
from pathlib import Path


def run_command(command, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100)
    print(" ".join(command))
    print(f"log: {log_path}")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    elapsed = time.time() - start
    if return_code != 0:
        raise RuntimeError(f"Command failed with code {return_code}: {' '.join(command)}")
    print(f"finished in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run baseline -> SFT -> RSFT -> DPO experiment.")
    parser.add_argument("--exp-dir", default="experiments/dpo_pipeline_20260618")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--math-subject", default="algebra")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--sft-steps", type=int, default=500)
    parser.add_argument("--rsft-rounds", type=int, default=32)
    parser.add_argument("--rsft-question-batch-size", type=int, default=4)
    parser.add_argument("--rsft-num-generations", type=int, default=4)
    parser.add_argument("--rsft-sft-steps-per-round", type=int, default=2)
    parser.add_argument("--dpo-steps", type=int, default=200)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    logs_dir = exp_dir / "logs"
    eval_dir = exp_dir / "eval"
    exp_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    config_path = exp_dir / "pipeline_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    sft_dir = exp_dir / "sft"
    rsft_dir = exp_dir / "rsft"
    dpo_dir = exp_dir / "dpo"

    eval_common = [
        "--backend",
        "transformers",
        "--dataset",
        "math",
        "--math-subject",
        args.math_subject,
        "--split",
        "test",
        "--max-samples",
        str(args.eval_samples),
        "--batch-size",
        "1",
        "--temperature",
        "0",
        "--max-tokens",
        "512",
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
        "--fast-grade",
        "--print-samples",
        "2",
    ]

    run_command(
        [
            "python",
            "prepare_baseline.py",
            "--model",
            args.model,
            "--output",
            str(eval_dir / "baseline_test.jsonl"),
            *eval_common,
        ],
        logs_dir / "eval_baseline.log",
    )

    run_command(
        [
            "python",
            "train_sft.py",
            "--model",
            args.model,
            "--dataset",
            "math",
            "--math-subject",
            args.math_subject,
            "--split",
            "train",
            "--max-samples",
            str(args.train_samples),
            "--eval-split",
            "train",
            "--eval-skip-samples",
            str(args.train_samples),
            "--eval-max-samples",
            "128",
            "--max-seq-length",
            str(args.max_seq_length),
            "--train-steps",
            str(args.sft_steps),
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "4",
            "--learning-rate",
            "1e-5",
            "--optimizer",
            "adamw",
            "--weight-decay",
            "0",
            "--max-grad-norm",
            "1.0",
            "--dtype",
            args.dtype,
            "--output-dir",
            str(sft_dir),
            "--log-every",
            "10",
            "--eval-every",
            "50",
            "--eval-batch-size",
            "1",
            "--save-every",
            "0",
            "--save-final",
        ],
        logs_dir / "train_sft.log",
    )

    run_command(
        [
            "python",
            "prepare_baseline.py",
            "--model",
            str(sft_dir / "step_final"),
            "--output",
            str(eval_dir / "sft_test.jsonl"),
            *eval_common,
        ],
        logs_dir / "eval_sft.log",
    )

    run_command(
        [
            "python",
            "train_rsft.py",
            "--model",
            str(sft_dir / "step_final"),
            "--dataset",
            "math",
            "--math-subject",
            args.math_subject,
            "--split",
            "train",
            "--max-samples",
            str(args.train_samples),
            "--max-seq-length",
            str(args.max_seq_length),
            "--rsft-rounds",
            str(args.rsft_rounds),
            "--question-batch-size",
            str(args.rsft_question_batch_size),
            "--num-generations",
            str(args.rsft_num_generations),
            "--generation-batch-size",
            "1",
            "--temperature",
            "1.0",
            "--top-p",
            "1.0",
            "--max-new-tokens",
            "512",
            "--min-new-tokens",
            "1",
            "--max-model-len",
            str(args.max_model_len),
            "--reward-threshold",
            "1.0",
            "--fast-grade",
            "--sft-steps-per-round",
            str(args.rsft_sft_steps_per_round),
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "4",
            "--learning-rate",
            "5e-6",
            "--optimizer",
            "adamw",
            "--max-grad-norm",
            "1.0",
            "--dtype",
            args.dtype,
            "--output-dir",
            str(rsft_dir),
            "--log-every",
            "10",
            "--eval-every-rounds",
            "8",
            "--eval-split",
            "train",
            "--eval-skip-samples",
            str(args.train_samples),
            "--eval-max-samples",
            "64",
            "--eval-batch-size",
            "1",
            "--save-every-rounds",
            "0",
            "--save-final",
        ],
        logs_dir / "train_rsft.log",
    )

    run_command(
        [
            "python",
            "prepare_baseline.py",
            "--model",
            str(rsft_dir / "step_final"),
            "--output",
            str(eval_dir / "rsft_test.jsonl"),
            *eval_common,
        ],
        logs_dir / "eval_rsft.log",
    )

    run_command(
        [
            "python",
            "train_dpo.py",
            "--model",
            str(rsft_dir / "step_final"),
            "--ref-model",
            str(rsft_dir / "step_final"),
            "--preference-file",
            str(rsft_dir / "samples.jsonl"),
            "--output-dir",
            str(dpo_dir),
            "--max-seq-length",
            str(args.max_seq_length),
            "--train-steps",
            str(args.dpo_steps),
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "4",
            "--learning-rate",
            "1e-6",
            "--beta",
            "0.1",
            "--max-grad-norm",
            "1.0",
            "--dtype",
            args.dtype,
            "--log-every",
            "10",
            "--save-final",
        ],
        logs_dir / "train_dpo.log",
    )

    run_command(
        [
            "python",
            "prepare_baseline.py",
            "--model",
            str(dpo_dir / "step_final"),
            "--output",
            str(eval_dir / "dpo_test.jsonl"),
            *eval_common,
        ],
        logs_dir / "eval_dpo.log",
    )

    print(f"\nPipeline complete: {exp_dir}")


if __name__ == "__main__":
    main()
