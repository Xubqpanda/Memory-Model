import torch.nn as nn

from ..config import ModelConfig


class TokenEmbedding(nn.Embedding):
    """Map token IDs to trainable hidden representations."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.vocab_size, config.d_model)


class PositionEmbedding(nn.Embedding):
    """Learned absolute position embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.block_size, config.d_model)
