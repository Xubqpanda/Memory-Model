import torch
import torch.nn.functional as F

from memory_model import ModelConfig
from memory_model.models import GELUFeedForward, SwiGLUFeedForward, TransformerLM


def ffn_config(ffn_type: str, d_ff: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        d_model=12,
        d_ff=d_ff,
        dropout=0.0,
        bias=False,
        ffn_type=ffn_type,
    )


def test_ffn_factory_selects_configured_variant():
    gelu_model = TransformerLM(ffn_config("gelu", 48))
    swiglu_model = TransformerLM(ffn_config("swiglu", 32))

    assert isinstance(gelu_model.blocks[0].ffn, GELUFeedForward)
    assert isinstance(swiglu_model.blocks[0].ffn, SwiGLUFeedForward)


def test_ffn_default_width_depends_on_the_variant():
    gelu = ModelConfig(d_model=12, n_head=2, ffn_type="gelu")
    swiglu = ModelConfig(d_model=12, n_head=2, ffn_type="swiglu")

    assert gelu.d_ff == 48
    assert swiglu.d_ff == 32


def test_swiglu_matches_the_gate_times_value_formula():
    config = ModelConfig(
        vocab_size=8,
        block_size=4,
        n_layer=1,
        n_head=1,
        d_model=2,
        d_ff=2,
        dropout=0.0,
        bias=False,
        ffn_type="swiglu",
    )
    ffn = SwiGLUFeedForward(config)
    with torch.no_grad():
        identity = torch.eye(2)
        ffn.gate_proj.weight.copy_(identity)
        ffn.up_proj.weight.copy_(identity)
        ffn.down_proj.weight.copy_(identity)

    x = torch.tensor([[[1.0, -1.0]]])
    expected = F.silu(x) * x
    torch.testing.assert_close(ffn(x), expected)


def test_swiglu_and_four_x_gelu_have_equal_ffn_parameter_counts():
    gelu = GELUFeedForward(ffn_config("gelu", 48))
    swiglu = SwiGLUFeedForward(ffn_config("swiglu", 32))

    gelu_parameters = sum(parameter.numel() for parameter in gelu.parameters())
    swiglu_parameters = sum(parameter.numel() for parameter in swiglu.parameters())
    assert gelu_parameters == swiglu_parameters == 1_152


def test_swiglu_gate_receives_gradients():
    model = TransformerLM(ffn_config("swiglu", 32))
    tokens = torch.randint(0, 32, (2, 6))
    output = model(tokens, targets=tokens)
    assert output.loss is not None
    output.loss.backward()

    assert model.blocks[0].ffn.gate_proj.weight.grad is not None
    assert model.blocks[0].ffn.up_proj.weight.grad is not None
    assert model.blocks[0].ffn.down_proj.weight.grad is not None
