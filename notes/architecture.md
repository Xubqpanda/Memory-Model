# `vanilla_transformer/` 架构详解：从 token 到下一个 token

本文对应项目中的模型实现：

- [`src/memory_model/models/vanilla_transformer/`](../src/memory_model/models/vanilla_transformer/)
- [`src/memory_model/config.py`](../src/memory_model/config.py)

`vanilla_transformer/` 是当前项目的标准 Transformer 基线架构包。它规定了模型内部有哪些层、数据如何流过这些层，以及训练和推理时模型返回什么。我们将组件拆成独立文件，是为了后续能够单独替换 Attention、FFN、Norm、Residual、Embedding 或输出头并进行消融实验，也为未来的 `memory_transformer/` 保留清楚的对照边界。

```text
vanilla_transformer/
├── attention.py    多头因果自注意力与 KV Cache
├── ffn.py          Position-wise FFN
├── embedding.py    Token 与位置 Embedding
├── norm.py         归一化组件
├── residual.py     残差连接策略
├── block.py        组装一个 Transformer Block
├── lm_head.py      语言模型输出头
├── transformer.py  组装完整 Decoder-only Transformer
├── types.py        KVCache 和 ModelOutput 类型
└── __init__.py     统一导出公共接口
```

但它并不负责完整训练流程。几个文件的职责分别是：

| 文件 | 职责 |
| --- | --- |
| `src/memory_model/models/vanilla_transformer/` | 定义并组装标准基线的可替换模型组件 |
| `src/memory_model/config.py` | 定义层数、隐藏维度、头数等规模参数 |
| `scripts/train/pretrain.py` | 读取数据、反向传播、更新参数、保存 checkpoint |
| `scripts/inference/generate.py` | 加载 checkpoint 并调用模型生成文本 |

## 1. 完整架构

当前模型是一个 Decoder-only Transformer，整体数据流为：

```text
input_ids
    │
    ├── Token Embedding
    ├── Position Embedding
    │
    ▼
初始 token 表征 X
    │
    ▼
Transformer Block 1
    │
    ▼
Transformer Block 2
    │
   ...
    │
    ▼
Transformer Block N
    │
    ▼
Final LayerNorm
    │
    ▼
LM Head
    │
    ▼
每个位置对整个词表的 logits
```

用公式表示：

$$
\text{token ids}
\rightarrow \text{token embedding}+\text{position embedding}
\rightarrow N\times\text{Transformer Block}
\rightarrow \text{LayerNorm}
\rightarrow \text{LM Head}
\rightarrow \text{next-token logits}
$$

## 2. 模型配置 `ModelConfig`

模型的规模由 `ModelConfig` 控制：

```python
@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    d_ff: int | None = None
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
```

各参数的含义是：

| 参数 | 含义 |
| --- | --- |
| `vocab_size` | 词表中一共有多少个 token |
| `block_size` | 模型一次最多处理多少个 token |
| `n_layer` | Transformer Block 的数量 |
| `n_head` | 每一层 Attention 的头数 |
| `d_model` | 每个 token 在模型中的表征维度 |
| `d_ff` | FFN 中间层维度，默认是 $4d_{model}$ |
| `dropout` | 训练时随机丢弃部分激活值的概率 |
| `bias` | Linear 和 LayerNorm 是否使用偏置参数 |
| `tie_embeddings` | 输入 Embedding 与输出 LM Head 是否共享权重 |

必须满足：

$$
d_{model}\bmod n_{head}=0
$$

因为每个 Attention Head 的维度是：

$$
d_{head}=\frac{d_{model}}{n_{head}}
$$

例如：

```text
d_model = 128
n_head  = 4
d_head  = 32
```

这并不是把 128 维扩大成 $4\times128$ 维，而是把原来的 128 维拆成 4 个 32 维的子空间。

## 3. 模型输入是什么

模型接收的 `input_ids` 是一个二维整数张量：

$$
\text{input\_ids.shape}=[B,T]
$$

其中：

- $B$ 是 batch size；
- $T$ 是 sequence length。

例如：

```text
input_ids = [
    [12, 56, 81, 20],
    [31, 42, 17, 90],
]
```

表示一个 batch 中有两段文本，每段有四个 token。整数本身只是词表索引，模型需要先把它转换成连续向量。

## 4. Token Embedding 与 Position Embedding

完整模型在 `TransformerLM.__init__` 中创建两个 Embedding：

```python
self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
self.position_embedding = nn.Embedding(config.block_size, config.d_model)
```

### 4.1 Token Embedding

Token Embedding 本质上是一个可训练矩阵：

$$
E_{token}\in\mathbb{R}^{V\times d_{model}}
$$

其中 $V$ 是 `vocab_size`。每个 token ID 会取出矩阵中的一行。

如果：

```text
vocab_size = 256
d_model    = 128
```

那么 Token Embedding 参数矩阵的形状就是：

```text
[256, 128]
```

### 4.2 Position Embedding

纯 Self-Attention 本身不知道 token 的先后顺序，所以还要加入位置向量：

$$
E_{position}\in\mathbb{R}^{T_{max}\times d_{model}}
$$

当前项目使用 learned absolute position embedding，即每个绝对位置都有一个可训练向量。

模型将两种向量直接相加：

```python
x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
```

因此：

$$
X_{b,t}=E_{token}(x_{b,t})+E_{position}(t)
$$

假设：

```text
batch_size     = 16
sequence_length = 64
d_model        = 128
```

那么 `x` 的形状为：

```text
[16, 64, 128]
```

意思是 16 段文本，每段 64 个 token，每个 token 当前由一个 128 维向量表示。

## 5. `CausalSelfAttention`

`CausalSelfAttention` 是 Transformer 相比传统 MLP 和 RNN 最关键的结构。

它的任务是：

> 让序列中的每个 token 根据当前上下文，有选择地读取其他 token 的信息。

### 5.1 一次生成 Q、K、V

代码中使用一个 Linear 同时生成 Q、K、V：

```python
self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
```

前向传播时：

```python
q, k, v = self.qkv(x).chunk(3, dim=-1)
```

数学上仍然等价于三个独立投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

这里合并成一次矩阵乘法只是为了提高计算效率。

三者可以从语义上理解为：

- Query：当前 token 想寻找什么信息；
- Key：当前 token 能用什么特征被其他 token 找到；
- Value：如果当前 token 被关注，实际提供什么信息。

假设输入形状为：

```text
x: [16, 64, 128]
```

QKV Linear 首先得到：

```text
qkv: [16, 64, 384]
```

切分以后：

```text
q: [16, 64, 128]
k: [16, 64, 128]
v: [16, 64, 128]
```

### 5.2 拆分多个 Attention Head

假设：

```text
n_head  = 4
d_model = 128
d_head  = 32
```

代码把 Q、K、V 从：

```text
[B, T, d_model]
```

变换为：

```text
[B, n_head, T, d_head]
```

具体例子是：

```text
[16, 64, 128]
        ↓ split_heads
[16, 4, 64, 32]
```

不同的 Head 没有被人工指定为“语法头”或“指代头”。每个 Head 的参数不同，它们在训练目标的驱动下自行学习适合的信息匹配方式。

### 5.3 Scaled Dot-Product Attention

Attention 的公式是：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}+M_{causal}
\right)V
$$

计算可以拆成四步：

1. $QK^\top$：计算 Query 与所有 Key 的匹配程度；
2. 除以 $\sqrt{d_k}$：控制数值尺度，避免 Softmax 过早饱和；
3. 加 causal mask：屏蔽未来 token；
4. 乘以 $V$：按注意力概率对信息进行加权汇总。

项目调用：

```python
F.scaled_dot_product_attention(...)
```

这是 PyTorch 对上述标准公式的高效实现，不是另一个不同的 Attention 算法。

### 5.4 为什么是 Causal Attention

语言模型在位置 $t$ 预测下一个 token 时，只能读取位置 $t$ 及其之前的信息：

```text
位置 0：只能看 0
位置 1：只能看 0, 1
位置 2：只能看 0, 1, 2
位置 3：只能看 0, 1, 2, 3
```

对应的允许访问矩阵是：

$$
M=
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
$$

如果训练时允许一个位置看到未来 token，模型就会直接偷看答案，训练目标也失去了意义。

### 5.5 合并多个 Head

各个 Head 计算完成后，张量形状仍为：

```text
[B, n_head, T, d_head]
```

代码重新排列并合并多个 Head：

```python
y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, width)
```

形状恢复为：

```text
[B, T, d_model]
```

最后经过：

```python
self.out_proj(y)
```

这个输出投影会重新混合不同 Head 得到的信息。

### 5.6 Attention Head 和 LM Head 不是同一个 Head

项目中会遇到两种完全不同的“头”：

| 名称 | 所属位置 | 数量 | 作用 |
| --- | --- | --- | --- |
| Attention Head | `attention.py` 的多头注意力内部 | 每层有 `n_head` 个 | 从不同表示子空间匹配并汇总上下文信息 |
| Language Model Head / LM Head | `lm_head.py`，模型最末端 | 整个模型通常只有一个 | 把隐藏表征映射成整个词表的 logits |

Attention Head 是 Attention 的一部分。假设 `d_model=128`、`n_head=4`，Attention 会把 Q、K、V 各自拆成 4 个 32 维的头：

```text
Q, K, V: [B, T, 128]
              ↓ 拆头
Q, K, V: [B, 4, T, 32]
```

我们没有为每个 Attention Head 创建一个独立的 Python 对象，而是把所有头放在同一个张量中并行计算。这种向量化实现更接近实际大模型，也更高效。不同头仍然对应 Q、K、V 投影矩阵中的不同参数切片，因此能够学到不同的信息匹配方式。

LM Head 则出现在所有 Transformer Block 之后：

```text
[B, T, d_model]
        ↓ LM Head
[B, T, vocab_size]
```

它不进行 token 之间的 Attention，也不存在 Query、Key、Value。这里叫 Head，是因为它是接在主干网络最后、为某个任务产生输出的“任务头”。如果主干用于分类，也可以接 Classification Head；当前任务是语言建模，所以接 Language Model Head。

## 6. `FeedForward`

FFN 的实现是：

```python
self.up_proj = nn.Linear(d_model, d_ff)
self.down_proj = nn.Linear(d_ff, d_model)
```

前向传播为：

```python
down_proj(GELU(up_proj(x)))
```

对应公式：

$$
\operatorname{FFN}(x)
=W_{down}\operatorname{GELU}(W_{up}x)
$$

假设：

```text
d_model = 128
d_ff    = 512
```

形状变化为：

```text
[B, T, 128]
      ↓ up_proj
[B, T, 512]
      ↓ GELU
[B, T, 512]
      ↓ down_proj
[B, T, 128]
```

Attention 和 FFN 的职责不同：

| 模块 | 主要职责 |
| --- | --- |
| Attention | 让不同 token 之间交换和汇总信息 |
| FFN | 对每个 token 当前拥有的表征进行非线性加工 |

FFN 对所有位置使用同一套权重，但每个位置独立计算。它不会直接让第 3 个 token 读取第 1 个 token；跨 token 的信息交换发生在 Attention 中。

## 7. `TransformerBlock`

一个 Block 包含四个组件：

```python
self.attn_norm = nn.LayerNorm(config.d_model)
self.attn = CausalSelfAttention(config)
self.ffn_norm = nn.LayerNorm(config.d_model)
self.ffn = FeedForward(config)
```

前向传播的核心只有两步：

```python
attn_out, new_cache = self.attn(self.attn_norm(x), ...)
x = x + attn_out

x = x + self.ffn(self.ffn_norm(x))
```

公式为：

$$
X'=X+\operatorname{Attention}(\operatorname{LayerNorm}(X))
$$

$$
Y=X'+\operatorname{FFN}(\operatorname{LayerNorm}(X'))
$$

这叫 Pre-Norm，因为 LayerNorm 位于 Attention 和 FFN 之前。

### 7.1 残差连接在做什么

以 Attention 子层为例：

```text
原始表征 X ─────────────────┐
    │                       │
    ▼                       │
LayerNorm                   │
    │                       │
    ▼                       │
Attention                   │
    │                       │
    └──────── 相加 ◀────────┘
                 │
                 ▼
                 X'
```

Attention 不必从零重新构造完整表征，而是学习一个增量：

$$
\Delta X=\operatorname{Attention}(\operatorname{LayerNorm}(X))
$$

然后：

$$
X'=X+\Delta X
$$

这和“直接学习梯度”不是一回事：

- 残差分支学习的是前向传播中的表征增量；
- 梯度是反向传播中损失对参数或中间变量的偏导数。

残差连接同时给前向信息和反向梯度提供了更直接的通道，使深层模型更容易训练。

## 8. 多层 Transformer

完整模型用下面的代码创建多个 Block：

```python
self.blocks = nn.ModuleList(
    [TransformerBlock(config) for _ in range(config.n_layer)]
)
```

如果 `n_layer=12`，就会创建 12 个结构相同但参数彼此独立的 Transformer Block：

```text
Embedding
    ↓
Block 1：自己的 Attention 和 FFN 参数
    ↓
Block 2：自己的 Attention 和 FFN 参数
    ↓
...
    ↓
Block 12：自己的 Attention 和 FFN 参数
```

因此，通常所说的“一个 12 层 Transformer”指 12 个 Transformer Block。每个 Block 内部都有一个 Attention 和一个 FFN，所以它同时也拥有 12 个 Attention 模块和 12 个 FFN 模块。

随着层数增加，每个 token 的表征可以被反复进行：

1. 从其他 token 汇总信息；
2. 对汇总后的信息进行非线性加工；
3. 将结果作为下一层的输入。

## 9. Final LayerNorm 与 LM Head

所有 Block 处理完成后，模型执行：

```python
logits = self.lm_head(self.final_norm(x))
```

`final_norm` 先稳定最后一层表征，然后 LM Head 把每个 token 的 $d_{model}$ 维向量投影到整个词表：

$$
\mathbb{R}^{d_{model}}\rightarrow\mathbb{R}^{V}
$$

假设：

```text
x.shape = [16, 64, 128]
vocab_size = 256
```

那么：

```text
logits.shape = [16, 64, 256]
```

含义是：

- 16 段文本；
- 每段有 64 个位置；
- 每个位置都对 256 个候选 token 给出一个分数。

Logits 还不是概率。对最后一个维度使用 Softmax 后，才得到候选 token 的概率分布：

$$
p(x_{t+1}=i\mid x_{\leq t})
=\operatorname{softmax}(z_t)_i
$$

### 9.1 Embedding Weight Tying

当 `tie_embeddings=True` 时：

```python
self.lm_head.weight = self.token_embedding.weight
```

输入 Embedding 与输出 LM Head 共用同一个参数矩阵。

这相当于：

- 输入时，用矩阵的一行把 token ID 转成语义向量；
- 输出时，用同一个矩阵衡量当前隐藏表征与各 token 表征的匹配程度。

它可以减少参数量，并经常改善语言模型训练效果。

## 10. Next-token Prediction 与 Loss

语言模型训练数据会将输入和目标错开一个位置：

```text
原始 token：Once  upon  a     time  there
输入 x：    Once  upon  a     time
目标 y：    upon  a     time  there
```

即：

$$
(x_1,x_2,\ldots,x_T)
\longrightarrow
(x_2,x_3,\ldots,x_{T+1})
$$

每个位置都执行一次 next-token prediction，而不只是最后一个位置：

```text
位置 0：根据 Once 预测 upon
位置 1：根据 Once upon 预测 a
位置 2：根据 Once upon a 预测 time
位置 3：根据 Once upon a time 预测 there
```

训练损失为所有位置交叉熵的平均值：

$$
\mathcal{L}
=-\frac{1}{BT}
\sum_{b=1}^{B}\sum_{t=1}^{T}
\log p(x_{b,t+1}\mid x_{b,\leq t})
$$

代码中：

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.reshape(-1),
)
```

这里把 batch 和时间维展平，是为了把所有位置统一交给交叉熵函数计算。

## 11. `model/` 与反向传播的关系

`model/transformer.py` 中的 `forward` 将各组件串联起来并建立完整计算图：

```text
Embedding
→ Attention
→ FFN
→ LM Head
→ Cross Entropy Loss
```

训练脚本随后执行：

```python
loss.backward()
optimizer.step()
```

`loss.backward()` 沿着这些模型组件建立的计算图反向计算所有参数的梯度，包括：

- Token Embedding；
- Position Embedding；
- 每一层的 Q、K、V 投影；
- 每一层的 Attention 输出投影；
- 每一层的 FFN 上投影和下投影；
- LayerNorm 参数；
- LM Head。

所有梯度会先根据同一次前向传播的参数状态计算完毕，然后 `optimizer.step()` 再统一更新参数。不会先更新最后一层，再用更新后的最后一层计算前面层的梯度。

## 12. KV Cache

自回归推理每次只生成一个新 token。如果完全不使用 Cache，那么每生成一步都要重新计算整段历史文本。

例如已经有：

```text
Once upon a time
```

第一次生成时，模型需要计算整段 prompt 的 Q、K、V。下一步生成时，旧 token 对应的 K、V 没有变化，因此可以把它们保存下来。

每个 Transformer Block 都拥有自己的 K 和 V，所以完整 Cache 的结构是：

```text
past_key_values = [
    Block 1 的 (K, V),
    Block 2 的 (K, V),
    ...
    Block N 的 (K, V),
]
```

生成新 token 时，只对新 token 计算新的 Q、K、V，再把新的 K、V 接到历史 Cache 后面：

```python
k = torch.cat((past_k, k), dim=2)
v = torch.cat((past_v, v), dim=2)
```

使用 Cache 后，每一步通常只需要把刚生成的一个 token 输入模型：

```python
model_input = generated[:, -1:]
```

需要注意：

- KV Cache 不是模型参数；
- KV Cache 不通过训练学习；
- 它是本次推理过程中产生的中间状态；
- 它不改变相同上下文下的模型输出；
- 它用更多显存换取更少的重复计算。

## 13. `ModelOutput`

模型返回一个 `ModelOutput`：

```python
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[KVCache] | None = None
```

三项结果分别是：

| 字段 | 使用场景 |
| --- | --- |
| `logits` | 训练和推理都需要，用来表示下一个 token 的候选分数 |
| `loss` | 传入 `targets` 时计算，训练时使用 |
| `past_key_values` | `use_cache=True` 时返回，自回归推理时使用 |

因此同一个 `forward` 同时支持训练和推理：

```text
训练：input_ids + targets → logits + loss
推理：input_ids + cache   → logits + new cache
```

## 14. 一次完整前向传播的形状变化

假设配置为：

```text
B          = 16
T          = 64
vocab_size = 256
d_model    = 128
n_head     = 4
d_head     = 32
d_ff       = 512
n_layer    = 2
```

完整形状变化如下：

| 阶段 | 张量形状 |
| --- | --- |
| `input_ids` | `[16, 64]` |
| Token Embedding | `[16, 64, 128]` |
| Position Embedding | `[1, 64, 128]`，广播到 batch |
| 相加后的 `x` | `[16, 64, 128]` |
| QKV 合并投影 | `[16, 64, 384]` |
| 单个 Q、K、V | `[16, 64, 128]` |
| 拆分多头后 | `[16, 4, 64, 32]` |
| Attention 合并多头后 | `[16, 64, 128]` |
| FFN 上投影 | `[16, 64, 512]` |
| FFN 下投影 | `[16, 64, 128]` |
| 两个 Block 后 | `[16, 64, 128]` |
| LM Head | `[16, 64, 256]` |
| Cross Entropy | 一个标量 loss |

一个重要规律是：

> 每个 Transformer Block 的输入和输出形状完全相同，都是 `[B, T, d_model]`。

只有这样，Block 才能不断堆叠，残差连接也才能直接执行张量相加。

## 15. 参数初始化

项目默认使用均值为 0、标准差为 0.02 的正态分布初始化 Linear 和 Embedding：

$$
W\sim\mathcal{N}(0,0.02^2)
$$

对于残差分支末端的 `out_proj.weight` 和 `down_proj.weight`，使用更小的标准差：

$$
\sigma=\frac{0.02}{\sqrt{2N}}
$$

其中 $N$ 是 Transformer Block 数量。

原因是深层网络会不断把 Attention 和 FFN 的输出加到残差流中。随着层数增加，适当缩小残差分支的初始输出，有助于控制网络初始化时的数值尺度。

## 16. 当前模型属于什么架构

当前实现可以描述为：

```text
Decoder-only Transformer
+ learned absolute position embedding
+ multi-head causal self-attention
+ Pre-LayerNorm
+ GELU FFN
+ residual connection
+ embedding weight tying
+ inference KV Cache
```

它是一个适合教学和实验的 GPT 风格基线，还不是现代 LLaMA 风格架构。后续可以逐项替换：

| 当前实现 | 后续可实验的实现 |
| --- | --- |
| Learned absolute position | RoPE |
| LayerNorm | RMSNorm |
| GELU FFN | SwiGLU |
| Multi-Head Attention | GQA / MQA |
| 标准 Attention 接口 | 显式研究 FlashAttention 与高效实现 |
| 无长期状态 | 参数化 memory 或外部 memory |

逐项替换的好处是：每次只改变一个变量，可以明确测量该结构对训练稳定性、计算成本、记忆能力和生成质量的影响。

## 17. 推荐阅读代码的顺序

第一次阅读 `model/` 时，推荐按照下面的顺序，而不是机械地按文件名阅读：

1. `block.py` 的 `TransformerBlock.forward`：先看一个 Block 的整体骨架；
2. `attention.py` 的 `CausalSelfAttention.forward`：理解 token 之间如何交换信息；
3. `ffn.py` 的 `FeedForward.forward`：理解单个 token 的非线性加工；
4. `transformer.py` 的 `TransformerLM.__init__`：看完整模型如何堆叠；
5. `transformer.py` 的 `TransformerLM.forward`：串起 Embedding、Blocks、LM Head 和 Loss；
6. `transformer.py` 的 `TransformerLM.generate`：理解自回归生成和 KV Cache。

掌握这条主线以后，我们后续加入 memory 时就可以准确讨论：memory 应该进入 Attention 之前、Attention 内部、Block 之间，还是作为独立的状态更新通道。
