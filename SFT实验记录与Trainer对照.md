# SFT 数据记录与 Trainer 对照实验

## 实验目的

本次实验按任务书要求，对 Qwen2.5-Math-1.5B 做监督微调 SFT，并比较训练前后性能。同时增加一个 Hugging Face `Trainer` 对照实验，用来验证手写 SFT 循环的行为是否与库训练器一致。

## 数据与设置

- 运行位置：租用服务器 RTX 4090 24GB。
- 模型：`Qwen/Qwen2.5-Math-1.5B`。
- 数据集：MATH algebra。
- 训练集：`train` split 前 256 条。
- 验证集：`train` split 跳过前 256 条后的 64 条，用于 teacher-forced response loss 曲线。
- 最终测试集：`test` split 前 64 条，用于生成评估。
- 最大序列长度：2048；本次训练和验证均未截断。
- 生成评估：`temperature=0`，`max_tokens=512`，`fast_grade=True`。
- 手写 SFT：全参数训练，response-only next-token cross entropy，AdamW，lr=1e-5，gradient_accumulation_steps=4，max_grad_norm=1.0。
- Trainer SFT：同样的数据构造、response-only loss、AdamW、constant lr=1e-5、gradient_accumulation_steps=4、max_grad_norm=1.0。

## 生成评估结果

| Model | Samples | Format Acc | Answer Acc | Reward |
|---|---:|---:|---:|---:|
| Base | 64 | 0.0000 | 0.0000 | 0.0000 |
| Handwritten SFT | 64 | 1.0000 | 0.7344 | 0.7344 |
| Trainer SFT | 64 | 1.0000 | 0.6875 | 0.6875 |

![Answer Accuracy](figures/generation_answer_accuracy.svg)

![Format Accuracy](figures/generation_format_accuracy.svg)

## 训练过程曲线

![Held-out Loss](figures/heldout_loss.svg)

![Held-out Token Accuracy](figures/heldout_token_accuracy.svg)

![Held-out Entropy](figures/heldout_entropy.svg)

![Training Loss](figures/train_loss.svg)

注意：`Training Loss` 图里的手写版 loss 与 Trainer 日志 loss 都来自各自训练循环的原始日志。由于 Trainer 内部对 gradient accumulation 的日志聚合方式不同，这张图只用于观察各自训练过程是否稳定；公平横向对比应主要看上面的 held-out response loss/token accuracy/entropy。

## 关键观察

1. Base 模型在当前 prompt 下不稳定输出 `</think> <answer>...</answer>` 格式，因此 test64 的 format/answer/reward 都是 0。
2. 手写 SFT 后格式准确率达到 1.0000，答案准确率达到 0.7344。
3. Trainer SFT 后格式准确率达到 1.0000，答案准确率达到 0.6875。
4. 两条训练曲线的 held-out loss 走势非常接近，说明手写 SFT 的核心训练逻辑与 Trainer 对照基本一致。
5. held-out loss 主要在前 50 到 150 step 明显下降，后续趋于平台期；这提示在 256 条训练样本上继续训练收益有限，并可能开始过拟合。
6. response entropy 随训练下降，说明模型在标准 response token 上更自信；这符合 SFT 的预期，但后续需要防止过度自信和泛化下降。

## 样例输出

### Base
**Sample 1**

Question: How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?

Response:

```text
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer>
```

Reward: `{'format_reward': 0.0, 'answer_reward': 0.0, 'reward': 0.0}`
**Sample 2**

Question: What is the positive difference between $120\%$ of 30 and $130\%$ of 20?

Response:

```text
The positive difference between $120\%$ of 30 and $130\%$ of 20 is 10.
```

Reward: `{'format_reward': 0.0, 'answer_reward': 0.0, 'reward': 0.0}`
### Handwritten SFT
**Sample 1**

Question: How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?

Response:

```text

There are two vertical asymptotes. The denominator factors as $(x+3)(x-2)$, so the vertical asymptotes are at $x=-3$ and $x=2$. </think> <answer>x=-3, x=2</answer>
```

Reward: `{'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}`
**Sample 2**

Question: What is the positive difference between $120\%$ of 30 and $130\%$ of 20?

Response:

```text

$120\%$ of 30 is $\frac{120}{100} \cdot 30 = 36$, and $130\%$ of 20 is $\frac{130}{100} \cdot 20 = 26$. The positive difference between 36 and 26 is $\boxed{10}$. </think> <answer>10</answer>
```

Reward: `{'format_reward': 1.0, 'answer_reward': 1.0, 'reward': 1.0}`
### Trainer SFT
**Sample 1**

Question: How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?

Response:

```text

The denominator of the function is $x^2+x-6=(x+3)(x-2)$.  Thus, the vertical asymptotes are at $x=-3$ and $x=2$, so the graph has $\boxed{2}$ vertical asymptotes. </think> <answer>2</answer>
```

Reward: `{'format_reward': 1.0, 'answer_reward': 1.0, 'reward': 1.0}`
**Sample 2**

Question: What is the positive difference between $120\%$ of 30 and $130\%$ of 20?

Response:

```text

$120\%$ of 30 is $\frac{120}{100} \cdot 30 = 36$, and $130\%$ of 20 is $\frac{130}{100} \cdot 20 = 26$. The positive difference between 36 and 26 is $\boxed{10}$. </think> <answer>10</answer>
```

Reward: `{'format_reward': 1.0, 'answer_reward': 1.0, 'reward': 1.0}`


## 文件说明

- `raw/hand/train_log.jsonl`：手写 SFT 训练日志。
- `raw/trainer/train_log.jsonl`：Trainer SFT 训练日志。
- `raw/eval/*.jsonl`：base、手写 SFT、Trainer SFT 的 test64 生成结果。
- `tables/*.csv`：从 JSONL 提取的曲线与指标表。
- `figures/*.svg`：报告中的图表。
