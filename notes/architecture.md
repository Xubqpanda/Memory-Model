# 从零实现 Decoder-only Transformer

这个项目把一个语言模型拆成下面这条数据流：

$$
\text{token ids}
\rightarrow \text{token embedding}+\text{position embedding}
\rightarrow N\times\text{Transformer Block}
\rightarrow \text{LayerNorm}
\rightarrow \text{LM Head}
\rightarrow \text{next-token logits}
$$

每个 Pre-Norm Transformer Block 是：

$$
X' = X + \operatorname{Attention}(\operatorname{LN}(X))
$$

$$
Y = X' + \operatorname{FFN}(\operatorname{LN}(X'))
$$

注意力中的投影为：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M_{\text{causal}}\right)V
$$

其中 causal mask 保证位置 $t$ 不能读取未来位置 $t+1,t+2,\ldots$。

FFN 对每一个 token 独立使用同一组参数：

$$
\operatorname{FFN}(x)=W_2\operatorname{GELU}(W_1x)
$$

Attention 负责 token 之间的信息交换，FFN 负责对每个 token 当前得到的表征进行非线性加工。残差连接让每个子层学习相对于输入的增量。

训练目标是 next-token prediction。输入和监督标签错开一个位置：

$$
(x_1,x_2,\ldots,x_{T})\longrightarrow(x_2,x_3,\ldots,x_{T+1})
$$

损失是所有位置的交叉熵平均值：

$$
\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log p(x_{t+1}\mid x_{\leq t})
$$

推理时，KV Cache 保存每一层历史 token 的 $K,V$。生成新 token 时只需为新 token 计算新的 $Q,K,V$，不必重复计算整段历史。Cache 不改变模型结果，只减少重复计算和增加显存占用。
