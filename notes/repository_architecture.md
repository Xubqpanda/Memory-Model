# LLM-Foundry 仓库组织架构

本文规定项目的代码边界和实验流程。目标是让 Transformer 底层架构、训练流程、记忆模块和知识编辑可以分别替换、测试和比较。

## 总体目录

```text
LLM-Foundry/
├── src/memory_model/                 可复用 Python 库
│   ├── models/                       模型架构组件
│   │   ├── attention/                MHA、GQA、MLA、KDA 等 Attention
│   │   ├── block/                    Transformer Block 组装
│   │   ├── embedding/                token 与 position embedding
│   │   ├── ffn/                      GELU、SwiGLU、MoE FFN
│   │   ├── lm_head/                  词表输出头
│   │   ├── memory/                   动态 memory 与读写规则
│   │   ├── norm/                     LayerNorm、RMSNorm
│   │   ├── residual/                 残差与 Residual Attention
│   │   └── transformer.py             Decoder-only Backbone
│   ├── data.py                       数据集与 loss mask
│   ├── tokenizer.py                  tokenizer 适配
│   ├── conversation.py               ChatML 和对话状态
│   ├── generation.py                 解码策略
│   ├── evaluation.py                 可复用评测指标
│   └── training/                     DDP、日志、偏好优化损失
├── configs/                          模型与训练超参数
├── scripts/
│   ├── data/                         下载、清洗、tokenize、二进制化
│   ├── train/                        pretrain、SFT、DPO 及 memory 训练
│   ├── inference/                    生成和网页推理
│   ├── eval/                         固定评测和 checkpoint 对比
│   └── memory/                       memory 轨迹生成和独立实验入口
├── data/                             本地数据，全部被 Git 忽略
├── checkpoints/                      本地 checkpoint，全部被 Git 忽略
├── logs/                             本地训练日志，全部被 Git 忽略
├── evals/                            固定评测数据、manifest 和评测结果
├── assets/                           可发布的小型静态资产，例如 tokenizer
├── tests/                            单元测试和最小行为测试
├── notes/                            原理、论文和实验设计笔记
└── pyproject.toml                    安装和测试配置
```

## 依赖方向

依赖应该从上到下流动：

```text
configs ───────────────┐
                       ▼
scripts ───────► src/memory_model ───────► PyTorch
   │                   │
   ├── data ───────────┘
   ├── train
   ├── inference
   └── eval  ─────────► evals/data
```

- `src/memory_model` 不应该 import `scripts`。
- `scripts` 只负责命令行参数、设备初始化、数据路径和训练流程编排。
- 模型组件通过 `ModelConfig` 构造，不在组件内部读取命令行参数。
- `tests` 优先直接测试 `src` 的公共行为，而不是启动完整训练脚本。

## 模型边界

`models/transformer.py` 是 Backbone 组装器，不应承载具体的 memory 或知识编辑算法。新的架构应遵循：

1. 具体机制放在独立子包，例如 `models/memory/metis.py`。
2. 通过配置或显式构造参数接入 `TransformerBlock`。
3. 保持 no-memory 基线路径不变。
4. 为状态生命周期、形状、梯度和 checkpoint 兼容性添加测试。

Memory 模块内部再分成四个职责：

```text
hidden states
    ├── selector       决定写哪些 token
    ├── projection     生成 memory key/value/query
    ├── writer         更新动态 memory state
    └── reader/fusion  读取并融合到 Backbone 残差流
```

## 实验产物边界

- 原始数据和 tokenized 数据放在 `data/`，不提交 Git。
- checkpoint 放在 `checkpoints/`，不提交 Git。
- 训练过程日志放在 `logs/` 或 `wandb/`，不提交 Git。
- 固定评测样本放在 `evals/data/`，可以提交；模型输出放在 `evals/results/`，不提交。
- 每个可复现实验应有一个 `configs/*.py` 和一个明确的 run name。

## Metis-lite 的第一阶段边界

第一阶段只实现可独立测试的动态 memory，不改变默认 Transformer 的 forward：

```text
models/memory/state.py       M_t、S_t 的状态容器
models/memory/selector.py    importance scorer 与 Alpha Top-P
models/memory/projection.py  K/V/Q 投影
models/memory/write.py       outer-product 与 gated delta rule
models/memory/read.py        query 读取和归一化
models/memory/fusion.py      memory 分支融合
models/memory/metis.py       组合上述组件
```

先在合成 Remember/Update/Forget 任务上验证，再通过 Transformer Block 的可选参数接入模型。这样每一次消融都可以回答一个清晰问题，而不会把现有预训练、SFT、DPO 基线和新的 memory 机制混在一起。
