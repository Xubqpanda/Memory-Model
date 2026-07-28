from dataclasses import asdict, dataclass


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

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

    def to_dict(self) -> dict:
        return asdict(self)
