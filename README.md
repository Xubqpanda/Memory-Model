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
- toy 数据和 TinyStories 数据准备
- 约 20M 和 125M 参数的实验配置

## 目录

```text
configs/               模型和训练超参数
src/                   Transformer 核心实现
scripts/data/          数据下载与预处理入口
scripts/train/         训练入口
scripts/inference/     推理与生成入口
tests/                 因果掩码、反向传播、KV Cache 等测试
notes/                 原理笔记
data/                  tokenized 数据，不提交到 Git
checkpoints/           模型检查点，不提交到 Git
```

## 1. 安装

当前环境已经具备依赖。若在其他环境运行：

```bash
cd /mnt/20t/xubuqiang/Study/Memory-Model
python -m pip install -e '.[data,dev]'
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

用 checkpoint 生成文本：

```bash
python scripts/inference/generate.py \
  --checkpoint checkpoints/tiny_debug/latest.pt \
  --prompt 'Once upon a time' \
  --max-new-tokens 100
```

## 3. TinyStories

先用少量故事验证下载和预处理流程：

```bash
python scripts/data/prepare_tinystories.py --limit-train 10000 --limit-val 1000
python scripts/train/pretrain.py --config configs/tinystories_20m.py --max-steps 100
```

确认无误后，删除 `data/tinystories_gpt2` 并处理完整数据：

```bash
python scripts/data/prepare_tinystories.py
python scripts/train/pretrain.py --config configs/tinystories_20m.py
```

125M 配置：

```bash
python scripts/train/pretrain.py --config configs/tinystories_125m.py
```

第一阶段建议先让 debug 模型明显降低 loss，再训练 20M。125M 配置用于第二阶段；它会消耗更多训练时间，不应在代码尚未验证时直接长跑。

## 阅读顺序

1. `src/tiny_transformer/model.py` 的 `TransformerBlock`
2. `CausalSelfAttention`
3. `TransformerLM.forward`
4. `TransformerLM.generate`
5. `scripts/train/pretrain.py`

公式版结构说明见 [`notes/architecture.md`](notes/architecture.md)。

## 研究路线

1. 建立可靠的 Decoder-only Transformer 与训练基线。
2. 在 TinyStories 等小规模数据上验证训练、生成和评测闭环。
3. 加入 memory 机制，并与基线进行参数量、计算量和建模能力对照。
4. 逐步研究时间编码、上下文修改历史和参数化长期状态。

项目仍处于早期阶段，实验接口会随着研究问题逐步演进。
