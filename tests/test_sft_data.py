import json

import numpy as np
import torch

from memory_model.data import SupervisedBinaryDataset


def test_supervised_dataset_shifts_mask_with_next_token_targets(tmp_path):
    metadata = {
        "sequence_length": 4,
        "train_examples": 1,
        "val_examples": 1,
    }
    (tmp_path / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    tokens = np.asarray([[10, 11, 12, 0]], dtype=np.uint16)
    mask = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)
    for split in ("train", "val"):
        tokens.tofile(tmp_path / f"{split}_tokens.bin")
        mask.tofile(tmp_path / f"{split}_loss_mask.bin")

    dataset = SupervisedBinaryDataset(tmp_path)
    x, y, supervised_tokens = dataset.get_batch("train", [0], torch.device("cpu"))

    assert x.tolist() == [[10, 11, 12]]
    assert y.tolist() == [[-100, 12, -100]]
    assert supervised_tokens == 1
