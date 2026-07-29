import json

import numpy as np
import torch

from memory_model.data import PreferenceBinaryDataset


def test_preference_dataset_returns_shifted_pairs_and_masks(tmp_path):
    metadata = {"sequence_length": 4, "train_examples": 1, "val_examples": 1}
    (tmp_path / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    sides = {
        "chosen": (
            np.asarray([[10, 11, 12, 0]], dtype=np.uint16),
            np.asarray([[0, 0, 1, 0]], dtype=np.uint8),
        ),
        "rejected": (
            np.asarray([[10, 11, 13, 0]], dtype=np.uint16),
            np.asarray([[0, 0, 1, 0]], dtype=np.uint8),
        ),
    }
    for split in ("train", "val"):
        for side, (tokens, mask) in sides.items():
            tokens.tofile(tmp_path / f"{split}_{side}_tokens.bin")
            mask.tofile(tmp_path / f"{split}_{side}_loss_mask.bin")

    dataset = PreferenceBinaryDataset(tmp_path)
    batch = dataset.get_batch("train", [0], torch.device("cpu"))

    assert batch["x_chosen"].tolist() == [[10, 11, 12]]
    assert batch["y_chosen"].tolist() == [[11, 12, 0]]
    assert batch["mask_chosen"].tolist() == [[False, True, False]]
    assert batch["y_rejected"].tolist() == [[11, 13, 0]]
