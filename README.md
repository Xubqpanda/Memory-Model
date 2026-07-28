# Memory-Model

这是一个从零理解并实验语言模型架构的研究项目。当前阶段先建立一个可验证的 Decoder-only Transformer 基线；后续将在这个基线上逐步实现和检验我们对参数化 memory、时间信息与有状态模型的想法。

模型核心由 PyTorch 基础算子手写，不调用 Hugging Face Transformers 的模型实现，因此每一项结构改动都可以被清楚地定位、比较和消融。

已经包含：

- learned token/position embedding
- causal multi-head self-attention
- Pre-Norm、残差连接、GELU FFN
- next-token cross-entropy loss
- embedding / LM head 权重共享
- greedy、temperature、top-k、top-p 解码
- 推理 KV Cache
- AdamW、梯度累积、梯度裁剪、warmup + cosine 学习率
- bf16、checkpoint 保存与恢复
- Weights & Biases 在线实验跟踪与断点续写
- toy 数据和 TinyStories 数据准备
- 约 20M 和 125M 参数的实验配置

## 目录

```text
configs/                                      模型和训练超参数
src/memory_model/                             项目级 Python 包
src/memory_model/models/vanilla_transformer/ 标准 Transformer 基线组件
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

125M 配置保留用于后续 MiniMind 中文基础模型实验：

```bash
python scripts/train/pretrain.py --config configs/tinystories_125m.py
```

第一阶段先完成 TinyStories 20M。第二阶段将数据入口切换为 MiniMind 中文预训练数据，再使用 125M 或更大配置继续 SFT、DPO 与 RL。

## 阅读顺序

1. `src/memory_model/models/vanilla_transformer/block.py` 的 `TransformerBlock`
2. `src/memory_model/models/vanilla_transformer/attention.py` 的 `CausalSelfAttention`
3. `src/memory_model/models/vanilla_transformer/transformer.py` 的 `TransformerLM.forward`
4. `src/memory_model/models/vanilla_transformer/transformer.py` 的 `TransformerLM.generate`
5. `scripts/train/pretrain.py`

公式版结构说明见 [`notes/architecture.md`](notes/architecture.md)。

## 研究路线

1. 建立可靠的 Decoder-only Transformer 与训练基线。
2. 在 TinyStories 等小规模数据上验证训练、生成和评测闭环。
3. 加入 memory 机制，并与基线进行参数量、计算量和建模能力对照。
4. 逐步研究时间编码、上下文修改历史和参数化长期状态。

项目仍处于早期阶段，实验接口会随着研究问题逐步演进。
