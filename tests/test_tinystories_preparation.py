import importlib.util

import numpy as np
import tiktoken


def load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_tinystories",
        "scripts/data/prepare_tinystories.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {"text": self.texts[item]}


def test_encode_split_batches_and_appends_eos(tmp_path):
    module = load_prepare_module()
    encoding = tiktoken.get_encoding("gpt2")
    dataset = FakeDataset(["A cat.", "A dog.", "The end."])
    output_path = tmp_path / "train.bin"

    token_count, story_count = module.encode_split(
        dataset,
        output_path,
        encoding,
        limit=2,
        batch_size=1,
        num_workers=1,
    )

    tokens = np.fromfile(output_path, dtype=np.uint16)
    expected = []
    for text in dataset.texts[:2]:
        expected.extend(encoding.encode_ordinary(text))
        expected.append(encoding.eot_token)

    assert story_count == 2
    assert token_count == len(expected)
    assert tokens.tolist() == expected
