"""Run one write/read cycle through the standalone Metis-lite memory."""

from __future__ import annotations

import argparse

import torch

from memory_model.models.memory import MetisLiteMemory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    memory = MetisLiteMemory(
        hidden_size=args.hidden_size,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
    )
    interaction = torch.randn(args.batch_size, args.sequence_length, args.hidden_size)

    written = memory(interaction)
    read_back = memory(interaction[:, :1], written.state, update_state=False)
    selected_count = written.selection.selected.sum(dim=-1).tolist()

    print(f"selected tokens per interaction: {selected_count}")
    print(f"update gate: {written.update_gate.tolist()}")
    print(f"state matrix shape: {tuple(written.state.matrix.shape)}")
    print(f"read output shape: {tuple(read_back.retrieved.shape)}")
    print(f"read output norm: {read_back.retrieved.norm().item():.6f}")


if __name__ == "__main__":
    main()

