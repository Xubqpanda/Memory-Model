import torch.nn as nn

from ....config import ModelConfig


class LearnedAbsolutePositionEmbedding(nn.Embedding):
    """Trainable lookup table with one vector for each absolute position.

    We follow the learned absolute position lookup used by GPT-style models.

    Reference:
        Radford et al., "Language Models are Unsupervised Multitask Learners"
        (GPT-2 technical report, 2019).
        https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.block_size, config.d_model)


# Backward-compatible name used in earlier notes and code.
PositionEmbedding = LearnedAbsolutePositionEmbedding
