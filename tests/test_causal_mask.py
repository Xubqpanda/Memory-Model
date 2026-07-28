import torch

from memory_model import ModelConfig
from memory_model.models.vanilla_transformer import TransformerLM


def test_future_tokens_cannot_change_past_logits():
    torch.manual_seed(0)
    model = TransformerLM(ModelConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, d_model=16)).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 9, 10]])
    logits_a = model(first).logits
    logits_b = model(changed_future).logits
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], atol=1e-6, rtol=1e-5)


def test_cached_and_full_forward_match():
    torch.manual_seed(0)
    model = TransformerLM(ModelConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, d_model=16)).eval()
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
