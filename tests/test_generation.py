import torch

from tiny_transformer.generation import sample_next_token


def test_greedy_selects_largest_logit():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    token = sample_next_token(logits, do_sample=False)
    assert token.tolist() == [[1]]


def test_top_k_one_is_deterministic():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    draws = [sample_next_token(logits, top_k=1).item() for _ in range(10)]
    assert draws == [1] * 10
