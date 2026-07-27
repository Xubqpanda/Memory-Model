from __future__ import annotations

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
