# Model components

模型代码按“可独立替换的研究组件”组织，不再放在统一的 `vanilla_transformer/` 目录中：

```text
models/
├── attention/
│   └── mha.py                         当前 MHA；后续加入 GQA、MLA、KDA
├── block/
│   └── transformer_block.py           组装 Pre-Norm Transformer Block
├── embedding/
│   ├── token_embedding/
│   │   └── learned.py                 可学习 token lookup table
│   └── position_embedding/
│       ├── learned_absolute.py        可学习绝对位置向量
│       └── rope.py                    Rotary Position Embedding
├── ffn/
│   └── gelu.py                        GELU FFN
├── lm_head/
│   └── language_model_head.py         next-token 输出头
├── norm/
│   └── layer_norm.py                  LayerNorm
├── residual/
│   └── standard.py                    标准加法残差；后续加入 AttnRes 等
├── transformer.py                     组装完整 Decoder-only LM
└── types.py                           KVCache 与 ModelOutput
```

Python 包和文件名统一使用小写；类名使用大驼峰。每个具体方法放在自己的文件中，并在类 docstring 中记录原论文、年份与链接。我们自己的改动必须明确标为项目实现或实验变体，不能冒充原论文方法。

## 当前方法与来源

| 组件 | 当前实现 | 主要来源 |
| --- | --- | --- |
| Token embedding | Learned lookup table | Bengio et al., [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html), 2003 |
| Absolute position | Learned absolute lookup table | Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019 |
| Rotary position | RoPE | Su et al., [RoFormer](https://arxiv.org/abs/2104.09864), 2021 |
| Attention | Multi-Head Attention | Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017 |
| FFN activation | GELU | Hendrycks and Gimpel, [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415), 2016 |
| Normalization | LayerNorm | Ba et al., [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016 |
| Residual | Additive residual | He et al., [Deep Residual Learning](https://arxiv.org/abs/1512.03385), 2015 |
| Block layout | Pre-LN Transformer | Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), 2020 |
| Weight tying | Input/output embedding sharing | Press and Wolf, [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859), 2016 |

## 位置编码配置

默认设置兼容已有 checkpoint：

```python
model = dict(
    attention_type="mha",
    position_embedding_type="learned_absolute",
)
```

新的 RoPE 实验使用：

```python
model = dict(
    attention_type="mha",
    position_embedding_type="rope",
    rope_theta=10_000.0,
)
```

Learned absolute position 在进入第一个 Block 前与 token embedding 相加。RoPE 不生成可训练的位置矩阵，而是在每一层 Attention 内旋转 Q 和 K。两者不是同一位置上的可互换权重，因此把旧 checkpoint 改成 RoPE 后需要重新训练。

`configs/minimind_pretrain_rope_60m.py` 是和原 60M MiniMind 预训练配置只相差位置编码方法的消融配置。
