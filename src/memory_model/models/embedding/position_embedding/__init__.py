from ....config import ModelConfig
from .learned_absolute import LearnedAbsolutePositionEmbedding, PositionEmbedding
from .rope import RotaryPositionEmbedding


def build_absolute_position_embedding(
    config: ModelConfig,
) -> LearnedAbsolutePositionEmbedding | None:
    """Build the residual-stream position module, if the method uses one."""

    if config.position_embedding_type == "learned_absolute":
        return LearnedAbsolutePositionEmbedding(config)
    if config.position_embedding_type == "rope":
        return None
    raise ValueError(f"unsupported position_embedding_type: {config.position_embedding_type}")


def build_rotary_position_embedding(config: ModelConfig) -> RotaryPositionEmbedding | None:
    """Build the attention-space position module, if the method uses one."""

    if config.position_embedding_type == "rope":
        return RotaryPositionEmbedding(config)
    if config.position_embedding_type == "learned_absolute":
        return None
    raise ValueError(f"unsupported position_embedding_type: {config.position_embedding_type}")


__all__ = [
    "LearnedAbsolutePositionEmbedding",
    "PositionEmbedding",
    "RotaryPositionEmbedding",
    "build_absolute_position_embedding",
    "build_rotary_position_embedding",
]
