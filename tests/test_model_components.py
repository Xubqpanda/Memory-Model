import torch

from memory_model import ModelConfig
from memory_model.models import (
    LanguageModelHead,
    ResidualConnection,
    TransformerLM,
)


def test_residual_connection_adds_sublayer_update():
    residual = torch.tensor([[1.0, 2.0]])
    update = torch.tensor([[0.5, -0.5]])
    result = ResidualConnection()(residual, update)
    torch.testing.assert_close(result, torch.tensor([[1.5, 1.5]]))


def test_language_model_head_is_not_an_attention_head():
    config = ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, d_model=16)
    model = TransformerLM(config)
    assert isinstance(model.lm_head, LanguageModelHead)
    assert model.lm_head.in_features == config.d_model
    assert model.lm_head.out_features == config.vocab_size
    assert model.blocks[0].attn.n_head == config.n_head
