import torch.nn as nn

from ...config import ModelConfig


class LanguageModelHead(nn.Linear):
    """Project hidden states to one next-token logit per vocabulary item.

    When ``tie_embeddings=True``, TransformerLM shares this matrix with the
    input token embedding.

    Reference for weight tying:
        Press and Wolf, "Using the Output Embedding to Improve Language Models"
        (2016). https://arxiv.org/abs/1608.05859
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.d_model, config.vocab_size, bias=False)
