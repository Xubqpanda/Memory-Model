# Memory-Model

这是一个从零理解并实验语言模型架构的研究项目。当前阶段先建立一个可验证的 Decoder-only Transformer 基线；后续将在这个基线上逐步实现和检验我们对参数化 memory、时间信息与有状态模型的想法。

模型核心由 PyTorch 基础算子手写，不调用 Hugging Face Transformers 的模型实现，因此每一项结构改动都可以被清楚地定位、比较和消融。

已经包含：

- learned token embedding，以及可切换的 learned absolute position / RoPE
- causal multi-head self-attention
- Pre-Norm、残差连接、GELU FFN
- next-token cross-entropy loss
- embedding / LM head 权重共享
- greedy、temperature、top-k、top-p 解码与 EOS 自动停止
- 推理 KV Cache
- AdamW、梯度累积、梯度裁剪、warmup + cosine 学习率
- bf16、checkpoint 保存与恢复
- Weights & Biases 在线实验跟踪与断点续写
- ChatML SFT 数据编码与 assistant-only loss mask
- 从预训练 checkpoint 初始化的双卡 DDP 全参数 SFT
- toy 数据和 TinyStories 数据准备
- 约 20M、60M 和 125M 参数的实验配置

## 目录

```text
configs/                                      模型和训练超参数
src/memory_model/                             项目级 Python 包
src/memory_model/models/                    可组合的 Transformer 模型组件
scripts/data/                                 数据下载与预处理入口
scripts/train/                                训练入口
scripts/inference/                            推理与生成入口
tests/                                        因果掩码、反向传播、KV Cache 等测试
notes/                                        原理笔记
data/                                         tokenized 数据，不提交到 Git
checkpoints/                                  模型检查点，不提交到 Git
```

## 1. 安装

当前环境已经具备依赖。若在其他环境运行：

```bash
cd /mnt/20t/xubuqiang/Study/Memory-Model
python -m pip install -e '.[data,dev]'
```

如需在线实验跟踪：

```bash
python -m pip install -e '.[data,dev,tracking]'
wandb login
```

不安装项目也可以直接运行 `scripts/` 中的入口。

## 2. 最小闭环

先生成无需联网的 byte-level toy 数据：

```bash
python scripts/data/prepare_toy.py
```

执行测试：

```bash
pytest -q
```

在 CPU 或 GPU 上跑几步：

```bash
python scripts/train/pretrain.py --config configs/tiny_debug.py --max-steps 20
```

可以为实验指定容易辨认的名称：

```bash
python scripts/train/pretrain.py \
  --config configs/tiny_debug.py \
  --wandb-run-name tiny-debug-baseline-200steps
```

默认会记录到 `Zjunlp-Xubqpanda/Memory-Model`。临时关闭或改为离线记录：

```bash
python scripts/train/pretrain.py \
  --config configs/tiny_debug.py \
  --max-steps 20 \
  --wandb-mode disabled

python scripts/train/pretrain.py \
  --config configs/tiny_debug.py \
  --max-steps 20 \
  --wandb-mode offline
```

W&B 本地运行日志统一保存在项目根目录的 `wandb/` 中，并由 `.gitignore` 排除。在线初始化失败时，训练会自动降级为离线记录；网络恢复后可以执行：

```bash
wandb sync wandb/offline-run-*
```

用 checkpoint 生成文本：

```bash
python scripts/inference/generate.py \
  --checkpoint checkpoints/tiny_debug/latest.pt \
  --prompt 'Once upon a time' \
  --max-new-tokens 100
```

## 3. TinyStories

TinyStories 完整训练集约有 212 万篇故事、约 4.45 亿 GPT-2 token，压缩下载量约 1GB。当前网络环境可以通过 Hugging Face 镜像准备数据：

```bash
cd /mnt/20t/xubuqiang/Study/Memory-Model

/mnt/8t/xubuqiang/anaconda3/bin/python \
  scripts/data/prepare_tinystories.py \
  --hf-endpoint https://hf-mirror.com \
  --batch-size 1000 \
  --num-workers 16
```

处理完成后会生成：

```text
data/tinystories_gpt2/
├── train.bin
├── val.bin
└── meta.json
```

正式训练 20M baseline。推荐使用两张 A100 通过 DDP 训练。W&B 直连可用，而本地 Git/Hugging Face 代理可能干扰 W&B，因此训练前显式取消代理：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

CUDA_VISIBLE_DEVICES=0,1 \
/mnt/8t/xubuqiang/anaconda3/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  scripts/train/pretrain.py \
  --config configs/tinystories_20m.py
```

当前配置每张 GPU 使用 batch 128。训练器会自动保持全局 batch 不变：

```text
单卡：128 × 256 × accumulation 2 = 65,536 token/step
双卡：2 × 128 × 256 × accumulation 1 = 65,536 token/step
```

因此可以从单卡 checkpoint 无缝切换到双卡 DDP，不需要改变学习率或训练步数。配置训练 8,000 steps，总计约 5.24 亿采样 token。终端 tqdm 会显示 loss、val loss、学习率、梯度、吞吐量和 ETA。

本地日志保存在：

```text
logs/tinystories-20m-baseline/<timestamp>/
├── train.log
├── metrics.jsonl
└── config.json
```

checkpoint 保存在：

```text
checkpoints/tinystories_20m/
├── latest.pt
└── best.pt
```

中断后继续训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/mnt/8t/xubuqiang/anaconda3/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  scripts/train/pretrain.py \
  --config configs/tinystories_20m.py \
  --resume checkpoints/tinystories_20m/latest.pt
```

如果只想先验证 20M 模型能否运行，可以准备较小数据并执行短训练：

```bash
/mnt/8t/xubuqiang/anaconda3/bin/python \
  scripts/data/prepare_tinystories.py \
  --hf-endpoint https://hf-mirror.com \
  --limit-train 10000 \
  --limit-val 1000

CUDA_VISIBLE_DEVICES=0 \
/mnt/8t/xubuqiang/anaconda3/bin/python \
  scripts/train/pretrain.py \
  --config configs/tinystories_20m.py \
  --max-steps 100
```

125M 配置保留用于后续更大规模实验：

```bash
python scripts/train/pretrain.py --config configs/tinystories_125m.py
```

TinyStories 20M 用于验证英文预训练闭环。中文预训练使用下一节的 MiniMind 60M baseline，后续再扩展到更大模型以及 SFT、DPO 与 RL。

## 4. MiniMind 中文预训练

当前使用 MiniMind 官方的 6,400 词表 tokenizer，资产保存在：

    assets/tokenizers/minimind/
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── chat_template.jinja

这里复用 MiniMind 的 tokenizer 和数据，但模型仍然是我们手写的 vanilla Transformer baseline：

    8 Transformer blocks
    8 attention heads
    d_model = 768
    d_ff = 3072
    learned absolute position embedding
    LayerNorm + GELU + MHA
    总参数量 = 62,141,184

它不是 MiniMind-3 官方的 Qwen3 风格模型；保留 vanilla 架构是为了后续进行清楚的结构消融。

将下载的原始文件放在：

    data/minimind/raw/pretrain_t2t_mini.jsonl

编码为训练可直接随机读取的 uint16 token 文件：

    /mnt/8t/xubuqiang/anaconda3/bin/python \
      scripts/data/prepare_minimind.py \
      --input data/minimind/raw/pretrain_t2t_mini.jsonl \
      --out-dir data/minimind/pretrain_mini \
      --validation-ratio 0.01 \
      --batch-size 4096

当前 mini 数据处理结果：

    总文档：      1,270,238
    训练文档：    1,257,552
    验证文档：       12,686
    训练 token： 327,910,695
    验证 token：   3,314,391

正式双卡预训练：

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

    CUDA_VISIBLE_DEVICES=0,1 \
    /mnt/8t/xubuqiang/anaconda3/bin/python \
      -m torch.distributed.run \
      --standalone \
      --nproc_per_node=2 \
      scripts/train/pretrain.py \
      --config configs/minimind_pretrain_60m.py

训练规模：

    2 GPU × 每卡 batch 128 × sequence 768 × accumulation 1
    = 196,608 token/step

    6,400 steps × 196,608 token
    = 1,258,291,200 sampled tokens
    ≈ 3.84 dataset epochs
    ≈ 20.2 training tokens/parameter

双卡 smoke test 的稳定训练吞吐约为 625K token/s，每卡 reserved 显存约 28.1GB。第一次 step 包含 torch.compile 编译时间，不代表后续训练速度。

中断后继续：

    CUDA_VISIBLE_DEVICES=0,1 \
    /mnt/8t/xubuqiang/anaconda3/bin/python \
      -m torch.distributed.run \
      --standalone \
      --nproc_per_node=2 \
      scripts/train/pretrain.py \
      --config configs/minimind_pretrain_60m.py \
      --resume checkpoints/minimind_pretrain_60m/latest.pt

## 5. MiniMind SFT

SFT 保留完整的 system/user/assistant/tool 上下文，但只对 assistant 正文及其
`<|im_end|>` 结束标记计算交叉熵。数据中的其他 token 会被转换成 `-100` 标签，
因此仍参与前向传播，却不会成为直接的预测目标。

先把 905,718 条 `sft_t2t_mini.jsonl` 对话编码成定长 token 和 loss-mask 文件：

```bash
/mnt/8t/xubuqiang/anaconda3/bin/python \
  scripts/data/prepare_minimind_sft.py \
  --input data/minimind/raw/sft_t2t_mini.jsonl \
  --out-dir data/minimind/sft_mini \
  --block-size 768 \
  --validation-ratio 0.01 \
  --batch-size 512
```

默认保留数据集已有的 `reasoning_content`，并按 MiniMind 的训练分布为 20% 的普通
回答保留空 `<think>` 标签。如果只训练直接回答，可以在编码时增加
`--exclude-reasoning --empty-think-ratio 0`。这一选择会写入 `meta.json`，便于实验追踪。

双卡正式 SFT：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

CUDA_VISIBLE_DEVICES=0,1 \
/mnt/8t/xubuqiang/anaconda3/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  scripts/train/sft.py \
  --config configs/minimind_sft_60m.py
```

默认配置从下面的预训练权重初始化，而不是重新随机初始化：

```text
checkpoints/minimind_pretrain_full_60m/best.pt
```

训练 2 个 epoch，每个 epoch 保存一个 checkpoint；同时维护 `latest.pt` 和验证集
assistant loss 最低的 `best.pt`。终端、`logs/` 和 W&B 都会记录 loss、验证 loss、
梯度范数、学习率、显存、input-token 吞吐和 supervised-token 吞吐。

中断后恢复时使用 SFT checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/mnt/8t/xubuqiang/anaconda3/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  scripts/train/sft.py \
  --config configs/minimind_sft_60m.py \
  --resume checkpoints/minimind_sft_60m/latest.pt
```

SFT 完成后启动 ChatML 多轮对话网页：

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/inference/web_chat.py \
  --checkpoint checkpoints/minimind_sft_60m/best.pt \
  --device cuda:0 \
  --server-name 0.0.0.0 \
  --port 7861
```

这里的 `cuda:0` 指 `CUDA_VISIBLE_DEVICES=1` 映射后的第一张可见 GPU，也就是物理 GPU 1。
网页会保留多轮 user/assistant 历史，并在超出 768-token 上下文前按完整轮次丢弃最早消息。

## 6. MiniMind DPO

DPO 从 SFT `best.pt` 初始化可训练 Policy，同时冻结同一个 SFT checkpoint 作为 Reference。
每条数据包含同一 Prompt 下的 chosen 和 rejected 回答，loss 只统计 Assistant token。

编码 17,166 个偏好对：

```bash
python scripts/data/prepare_minimind_dpo.py \
  --input data/minimind/raw/dpo.jsonl \
  --out-dir data/minimind/dpo \
  --block-size 768 \
  --validation-ratio 0.05 \
  --batch-size 512
```

当前处理结果：

```text
训练偏好对：16,186
验证偏好对：850
chosen 截断：2,510
rejected 截断：2,252
```

双卡正式 DPO：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  scripts/train/dpo.py \
  --config configs/minimind_dpo_60m.py
```

默认配置使用每卡 32 个 preference pairs、全局 batch 64、`beta=0.1`、学习率 `5e-7`，
训练 1 个 epoch，约 253 optimizer steps。DPO 训练默认关闭 dropout 和 torch.compile，
确保 Policy 与 Reference 的 log-probability 数值路径一致。

DPO checkpoint 可直接通过同一个网页测试：

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/inference/web_chat.py \
  --checkpoint checkpoints/minimind_dpo_60m/best.pt \
  --device cuda:0 \
  --server-name 0.0.0.0 \
  --port 7862
```

## 7. 固定评测

永久保留的行为评测集位于：

```text
evals/data/fixed_chat_eval.jsonl
```

它包含 100 条不参与任何训练的测试样本，覆盖数学、知识、指令遵循、概念解释、摘要、多轮、
重复压力和 EOS/角色边界。使用相同解码设置比较 SFT 与 DPO：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/eval/compare_checkpoints.py \
  --checkpoint sft=checkpoints/minimind_sft_60m/best.pt \
  --checkpoint dpo=checkpoints/minimind_dpo_60m/best.pt \
  --greedy
```

结果写入 `evals/results/`，包括每条回答和汇总指标：

- 自动检查正确率；
- EOS Stop Rate；
- Role Leakage；
- 平均长度；
- Distinct-1/2/3 和重复率；
- 按类别统计。

`agent_rl_math.jsonl` 在开始 GRPO 前通过固定 seed 切分为 18,000/1,000/1,000：

```bash
python scripts/data/split_agent_rl_math.py
```

最终 `test.jsonl` 位于 `evals/data/agent_rl_math_split/`，被目录级 `.gitignore` 忽略，
只能在本地用于最终测试，不能用于训练或调参。切分 manifest 保存了 seed 和原始文件 SHA256。

训练完成后生成中文：

    /mnt/8t/xubuqiang/anaconda3/bin/python \
      scripts/inference/generate.py \
      --checkpoint checkpoints/minimind_pretrain_60m/best.pt \
      --prompt "人工智能的发展" \
      --max-new-tokens 200 \
      --temperature 0.8 \
      --top-k 50 \
      --top-p 0.95 \
      --device cuda:0

启动纯预训练续写网页：

    CUDA_VISIBLE_DEVICES=0 \
    /mnt/8t/xubuqiang/anaconda3/bin/python \
      scripts/inference/web_completion.py \
      --checkpoint checkpoints/minimind_pretrain_full_60m/best.pt \
      --device cuda:0 \
      --server-name 0.0.0.0 \
      --port 7860

浏览器打开 http://127.0.0.1:7860。该 harness 不注入 user、assistant 或 system role，只把模型续写原样追加到累计文本；超过 768 token 时从最早位置裁剪。它用于测试 base pretrained model 的纯文本 continuation 能力。

### MiniMind 完整预训练集

完整 pretrain_t2t 数据编码命令：

    /mnt/8t/xubuqiang/anaconda3/bin/python \
      scripts/data/prepare_minimind.py \
      --input data/minimind/raw/pretrain_t2t.jsonl \
      --out-dir data/minimind/pretrain_full \
      --validation-ratio 0.01 \
      --batch-size 8192

当前完整数据统计：

    总文档：        8,468,827
    训练文档：      8,384,378
    验证文档：         84,449
    训练 token： 2,186,853,770
    验证 token：    21,998,827

完整数据配置保持与 mini 实验完全相同的 62M 模型和全局 batch，从头训练 5 个数据 epoch：

    55,615 steps × 196,608 token/step
    = 10,934,353,920 sampled tokens
    ≈ 5.000039 dataset epochs

训练器在每个 epoch 边界强制验证并保存独立 checkpoint：

    epoch 1：completed step 11,123 → epoch_1.pt
    epoch 2：completed step 22,246 → epoch_2.pt
    epoch 3：completed step 33,369 → epoch_3.pt
    epoch 4：completed step 44,492 → epoch_4.pt
    epoch 5：completed step 55,615 → epoch_5.pt

正式双卡训练：

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

    CUDA_VISIBLE_DEVICES=0,1 \
    /mnt/8t/xubuqiang/anaconda3/bin/python \
      -m torch.distributed.run \
      --standalone \
      --nproc_per_node=2 \
      scripts/train/pretrain.py \
      --config configs/minimind_pretrain_full_60m.py

## 阅读顺序

1. `src/memory_model/models/block/transformer_block.py` 的 `TransformerBlock`
2. `src/memory_model/models/attention/mha.py` 的 `MultiHeadCausalSelfAttention`
3. `src/memory_model/models/transformer.py` 的 `TransformerLM.forward`
4. `src/memory_model/models/transformer.py` 的 `TransformerLM.generate`
5. `scripts/train/pretrain.py`

公式版结构说明见 [`notes/architecture.md`](notes/architecture.md)。
组件目录、扩展位置和论文来源见 [`src/memory_model/models/README.md`](src/memory_model/models/README.md)。

## 研究路线

1. 建立可靠的 Decoder-only Transformer 与训练基线。
2. 在 TinyStories 等小规模数据上验证训练、生成和评测闭环。
3. 加入 memory 机制，并与基线进行参数量、计算量和建模能力对照。
4. 逐步研究时间编码、上下文修改历史和参数化长期状态。

项目仍处于早期阶段，实验接口会随着研究问题逐步演进。
