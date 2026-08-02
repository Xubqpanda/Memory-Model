from dataclasses import asdict, dataclass
from typing import Literal


AttentionType = Literal["mha"]
FFNType = Literal["gelu", "swiglu"]
PositionEmbeddingType = Literal["learned_absolute", "rope"]


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
    attention_type: AttentionType = "mha"
    ffn_type: FFNType = "gelu"
    position_embedding_type: PositionEmbeddingType = "learned_absolute"
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.attention_type != "mha":
            raise ValueError(f"unsupported attention_type: {self.attention_type}")
        if self.ffn_type not in {"gelu", "swiglu"}:
            raise ValueError(f"unsupported ffn_type: {self.ffn_type}")
        if self.position_embedding_type not in {"learned_absolute", "rope"}:
            raise ValueError(
                f"unsupported position_embedding_type: {self.position_embedding_type}"
            )
        if self.position_embedding_type == "rope" and (self.d_model // self.n_head) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.d_ff is None:
            self.d_ff = (
                4 * self.d_model
                if self.ffn_type == "gelu"
                else (8 * self.d_model) // 3
            )
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    def matches(self, values: dict) -> bool:
        """Compare configs after filling defaults added in newer code versions."""

        try:
            return self.to_dict() == type(self)(**values).to_dict()
        except (TypeError, ValueError):
            return False
