import torch

from memory_model.models.memory import AlphaTopPSelector, MemoryReader, MemoryState, MetisLiteMemory


def test_alpha_top_p_selects_smallest_probability_prefix() -> None:
    selector = AlphaTopPSelector(hidden_size=2, alpha_top_p=0.8, straight_through=False)
    with torch.no_grad():
        selector.pool_score.weight.copy_(torch.tensor([[1.0, 0.0]]))

    hidden_states = torch.tensor([[[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]])
    output = selector(hidden_states)

    assert output.selected.tolist() == [[True, True, False, False]]
    assert torch.allclose(output.weights.sum(dim=-1), torch.ones(1))
    assert torch.all(output.weights[:, 2:] == 0)


def test_alpha_top_p_respects_padding_mask() -> None:
    selector = AlphaTopPSelector(hidden_size=2, alpha_top_p=1.0, straight_through=False)
    hidden_states = torch.randn(1, 4, 2)
    valid_mask = torch.tensor([[True, True, False, False]])
    output = selector(hidden_states, valid_mask)

    assert not output.selected[:, 2:].any()
    assert torch.all(output.weights[:, 2:] == 0)


def test_memory_reader_recovers_single_associative_pair() -> None:
    key = torch.tensor([[[1.0, -2.0, 0.5]]])
    value = torch.tensor([[[2.0, -1.0]]])
    matrix = torch.einsum("btk,btv->bkv", key, value)
    state = MemoryState(matrix=matrix, normalizer=key.abs().squeeze(1))
    retrieved = MemoryReader()(key, state)

    assert torch.allclose(retrieved, value, atol=1e-5)


def test_metis_lite_reads_previous_state_and_backpropagates() -> None:
    torch.manual_seed(0)
    memory = MetisLiteMemory(hidden_size=8, key_dim=4, value_dim=6)
    hidden_states = torch.randn(2, 5, 8, requires_grad=True)

    first = memory(hidden_states)
    second = memory(hidden_states, first.state)
    assert first.retrieved.shape == (2, 5, 6)
    assert second.fused.shape == hidden_states.shape
    assert first.state.matrix.shape == (2, 4, 6)
    assert torch.isfinite(second.retrieved).all()

    loss = second.retrieved.square().mean() + second.fused.square().mean()
    loss.backward()
    assert hidden_states.grad is not None
    assert memory.selector.pool_score.weight.grad is not None
