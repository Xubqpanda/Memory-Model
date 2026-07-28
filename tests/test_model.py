import torch

from memory_model import ModelConfig
from memory_model.models.vanilla_transformer import TransformerLM


def small_model() -> TransformerLM:
    torch.manual_seed(0)
    return TransformerLM(ModelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, d_model=16, dropout=0.0))


def test_forward_shape_and_backward():
    model = small_model()
    x = torch.randint(0, 32, (3, 8))
    y = torch.randint(0, 32, (3, 8))
    output = model(x, targets=y)
    assert output.logits.shape == (3, 8, 32)
    assert output.loss is not None
    output.loss.backward()
    assert model.blocks[0].attn.qkv.weight.grad is not None


def test_weight_tying():
    model = small_model()
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
