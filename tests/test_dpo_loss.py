import math

import torch

from memory_model import ModelConfig
from memory_model.models import TransformerLM
from memory_model.training import disable_model_dropout, dpo_loss, sequence_log_probs


def test_identical_policy_and_reference_start_at_log_two():
    chosen = torch.tensor([-3.0, -5.0])
    rejected = torch.tensor([-4.0, -6.0])
    output = dpo_loss(chosen, rejected, chosen, rejected, beta=0.1)

    assert torch.isclose(output.loss, torch.tensor(math.log(2.0)))
    assert output.preference_accuracy.item() == 0.0
    assert torch.equal(output.reward_margins, torch.zeros_like(chosen))


def test_dpo_rewards_policy_that_improves_chosen_relative_to_rejected():
    output = dpo_loss(
        policy_chosen_logps=torch.tensor([-2.0]),
        policy_rejected_logps=torch.tensor([-5.0]),
        reference_chosen_logps=torch.tensor([-3.0]),
        reference_rejected_logps=torch.tensor([-4.0]),
        beta=0.5,
    )

    assert output.loss.item() < math.log(2.0)
    assert output.preference_accuracy.item() == 1.0
    assert output.reward_margins.item() > 0


def test_sequence_log_probs_only_sum_masked_tokens():
    logits = torch.tensor(
        [
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ]
    )
    labels = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[False, True, True]])

    total = sequence_log_probs(logits, labels, mask)
    average = sequence_log_probs(logits, labels, mask, average=True)

    assert torch.isclose(total, 2 * average).all()


def test_disabling_dropout_makes_train_and_eval_forward_identical():
    model = TransformerLM(
        ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, d_model=8, dropout=0.5)
    )
    disable_model_dropout(model)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    model.train()
    train_logits = model(input_ids).logits
    model.eval()
    eval_logits = model(input_ids).logits

    assert torch.equal(train_logits, eval_logits)
