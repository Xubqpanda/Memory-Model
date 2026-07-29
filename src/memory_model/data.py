from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class BinaryTokenDataset:
    """Random next-token batches from uint16 token files."""

    def __init__(self, data_dir: str | Path, block_size: int) -> None:
        data_dir = Path(data_dir)
        self.block_size = block_size
        self.train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    def get_batch(self, split: str, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        if len(data) <= self.block_size:
            raise ValueError(f"{split}.bin needs more than block_size={self.block_size} tokens")
        starts = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(np.asarray(data[i : i + self.block_size], dtype=np.int64)) for i in starts]
        )
        y = torch.stack(
            [torch.from_numpy(np.asarray(data[i + 1 : i + 1 + self.block_size], dtype=np.int64)) for i in starts]
        )
        pin = device.type == "cuda"
        return (
            x.pin_memory().to(device, non_blocking=pin),
            y.pin_memory().to(device, non_blocking=pin),
        )


class SupervisedBinaryDataset:
    """Fixed-length SFT examples with a token-aligned assistant loss mask."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.metadata = json.loads((self.data_dir / "meta.json").read_text(encoding="utf-8"))
        self.sequence_length = int(self.metadata["sequence_length"])
        self.block_size = self.sequence_length - 1
        self._tokens: dict[str, np.memmap] = {}
        self._masks: dict[str, np.memmap] = {}
        self._example_counts: dict[str, int] = {}

        for split in ("train", "val"):
            count = int(self.metadata[f"{split}_examples"])
            self._example_counts[split] = count
            self._tokens[split] = np.memmap(
                self.data_dir / f"{split}_tokens.bin",
                dtype=np.uint16,
                mode="r",
                shape=(count, self.sequence_length),
            )
            self._masks[split] = np.memmap(
                self.data_dir / f"{split}_loss_mask.bin",
                dtype=np.uint8,
                mode="r",
                shape=(count, self.sequence_length),
            )

    def num_examples(self, split: str) -> int:
        return self._example_counts[split]

    def get_batch(
        self,
        split: str,
        indices: np.ndarray | torch.Tensor | list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices, dtype=np.int64)
        rows = np.asarray(self._tokens[split][indices], dtype=np.int64)
        masks = np.asarray(self._masks[split][indices], dtype=np.bool_)

        x = torch.from_numpy(rows[:, :-1].copy())
        y = torch.from_numpy(rows[:, 1:].copy())
        target_mask = torch.from_numpy(masks[:, 1:].copy())
        y.masked_fill_(~target_mask, -100)
        supervised_tokens = int(target_mask.sum().item())

        pin = device.type == "cuda"
        if pin:
            x = x.pin_memory()
            y = y.pin_memory()
        return (
            x.to(device, non_blocking=pin),
            y.to(device, non_blocking=pin),
            supervised_tokens,
        )


class PreferenceBinaryDataset:
    """Fixed-length chosen/rejected pairs for preference optimization."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.metadata = json.loads((self.data_dir / "meta.json").read_text(encoding="utf-8"))
        self.sequence_length = int(self.metadata["sequence_length"])
        self.block_size = self.sequence_length - 1
        self._tokens: dict[tuple[str, str], np.memmap] = {}
        self._masks: dict[tuple[str, str], np.memmap] = {}
        self._example_counts: dict[str, int] = {}

        for split in ("train", "val"):
            count = int(self.metadata[f"{split}_examples"])
            self._example_counts[split] = count
            for side in ("chosen", "rejected"):
                key = (split, side)
                self._tokens[key] = np.memmap(
                    self.data_dir / f"{split}_{side}_tokens.bin",
                    dtype=np.uint16,
                    mode="r",
                    shape=(count, self.sequence_length),
                )
                self._masks[key] = np.memmap(
                    self.data_dir / f"{split}_{side}_loss_mask.bin",
                    dtype=np.uint8,
                    mode="r",
                    shape=(count, self.sequence_length),
                )

    def num_examples(self, split: str) -> int:
        return self._example_counts[split]

    def get_batch(
        self,
        split: str,
        indices: np.ndarray | torch.Tensor | list[int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices, dtype=np.int64)
        pin = device.type == "cuda"
        batch: dict[str, torch.Tensor] = {}

        for side in ("chosen", "rejected"):
            rows = np.asarray(self._tokens[(split, side)][indices], dtype=np.int64)
            masks = np.asarray(self._masks[(split, side)][indices], dtype=np.bool_)
            x = torch.from_numpy(rows[:, :-1].copy())
            y = torch.from_numpy(rows[:, 1:].copy())
            target_mask = torch.from_numpy(masks[:, 1:].copy())
            if pin:
                x = x.pin_memory()
                y = y.pin_memory()
                target_mask = target_mask.pin_memory()
            batch[f"x_{side}"] = x.to(device, non_blocking=pin)
            batch[f"y_{side}"] = y.to(device, non_blocking=pin)
            batch[f"mask_{side}"] = target_mask.to(device, non_blocking=pin)
        return batch
