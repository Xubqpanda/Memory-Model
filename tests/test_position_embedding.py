import pytest
import torch

from memory_model import ModelConfig
from memory_model.models import (
    LearnedAbsolutePositionEmbedding,
    RotaryPositionEmbedding,
    TransformerLM,
)


def position_config(position_embedding_type: str) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        d_model=16,
        dropout=0.0,
        position_embedding_type=position_embedding_type,
    )


def test_learned_absolute_position_embedding_is_a_trainable_lookup_table():
    config = position_config("learned_absolute")
    model = TransformerLM(config)

    assert isinstance(model.position_embedding, LearnedAbsolutePositionEmbedding)
    assert model.position_embedding.weight.shape == (config.block_size, config.d_model)

    model(torch.randint(0, config.vocab_size, (2, 8))).logits.sum().backward()
    assert model.position_embedding.weight.grad is not None


def test_rope_rotates_queries_and_keys_without_trainable_position_parameters():
    config = position_config("rope")
    model = TransformerLM(config)
    rope = RotaryPositionEmbedding(config)
    query = torch.randn(1, config.n_head, 3, config.d_model // config.n_head)
    key = torch.randn_like(query)

    rotated_query, rotated_key = rope(query, key, torch.arange(3))

    assert model.position_embedding is None
    assert not any("position_embedding" in name for name, _ in model.named_parameters())
    torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
    torch.testing.assert_close(rotated_key[:, :, 0], key[:, :, 0])
    assert not torch.equal(rotated_query[:, :, 1:], query[:, :, 1:])
    assert not torch.equal(rotated_key[:, :, 1:], key[:, :, 1:])


def test_rope_cached_and_full_forward_match():
    torch.manual_seed(0)
    model = TransformerLM(position_config("rope")).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    full_logits = model(tokens).logits
    cache = None
    pieces = []

    for index in range(tokens.size(1)):
        output = model(tokens[:, index : index + 1], past_key_values=cache, use_cache=True)
        pieces.append(output.logits)
        cache = output.past_key_values

    cached_logits = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full_logits, cached_logits, atol=1e-5, rtol=1e-5)


def test_rope_requires_even_head_dimension():
    with pytest.raises(ValueError, match="even attention head dimension"):
        ModelConfig(
            vocab_size=32,
            block_size=16,
            n_layer=1,
            n_head=2,
            d_model=10,
            position_embedding_type="rope",
        )
