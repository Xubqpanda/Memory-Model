import torch

from memory_model import ModelConfig
from memory_model.models import TransformerLM


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


def test_config_matches_checkpoint_created_before_position_options_existed():
    current = ModelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, d_model=16)
    legacy = current.to_dict()
    legacy.pop("attention_type")
    legacy.pop("position_embedding_type")
    legacy.pop("rope_theta")

    assert current.matches(legacy)
    assert not ModelConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        d_model=16,
        position_embedding_type="rope",
    ).matches(legacy)


def test_generation_stops_after_eos_token():
    config = ModelConfig(
        vocab_size=8,
        block_size=8,
        n_layer=1,
        n_head=1,
        d_model=8,
        dropout=0.0,
        tie_embeddings=False,
    )
    model = TransformerLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    prompt = torch.tensor([[1, 2]])
    generated = model.generate(
        prompt,
        max_new_tokens=4,
        do_sample=False,
        eos_token_id=0,
    )

    assert generated.tolist() == [[1, 2, 0]]
