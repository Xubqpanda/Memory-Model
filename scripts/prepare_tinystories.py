#!/usr/bin/env python3
"""Download TinyStories from Hugging Face and encode it into uint16 token files."""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
import tiktoken


def encode_split(dataset, output_path: Path, encoding, limit: int | None) -> int:
    eos = encoding.eot_token
    count = 0
    with output_path.open("wb") as output:
        for row_idx, row in enumerate(dataset):
            if limit is not None and row_idx >= limit:
                break
            ids = encoding.encode_ordinary(row["text"])
            ids.append(eos)
            array = np.asarray(ids, dtype=np.uint16)
            array.tofile(output)
            count += len(ids)
            if (row_idx + 1) % 10000 == 0:
                print(f"{output_path.name}: {row_idx + 1:,} stories, {count:,} tokens")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/tinystories_gpt2")
    parser.add_argument("--limit-train", type=int, default=None, help="Useful for a small trial download")
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoding = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("roneneldan/TinyStories")
    train_tokens = encode_split(dataset["train"], out_dir / "train.bin", encoding, args.limit_train)
    val_tokens = encode_split(dataset["validation"], out_dir / "val.bin", encoding, args.limit_val)
    metadata = {
        "dataset": "roneneldan/TinyStories",
        "tokenizer": "gpt2",
        "vocab_size": encoding.n_vocab,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }
    (out_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata)


if __name__ == "__main__":
    main()
