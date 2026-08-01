import torch.nn as nn

from ....config import ModelConfig


class TokenEmbedding(nn.Embedding):
    """Map token IDs to trainable hidden representations.

    Reference:
        Bengio et al., "A Neural Probabilistic Language Model" (2003).
        https://www.jmlr.org/papers/v3/bengio03a.html
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.vocab_size, config.d_model)
