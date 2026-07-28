import torch.nn as nn

from ...config import ModelConfig


class LanguageModelHead(nn.Linear):
    """Project hidden states to one logit per vocabulary token.

    This output head is unrelated to the individual heads inside multi-head
    attention. Attention heads mix contextual information; the LM head predicts
    the next token.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.d_model, config.vocab_size, bias=False)
