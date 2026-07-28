#!/usr/bin/env python3
"""Download TinyStories and encode it into uint16 GPT-2 token files."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tiktoken
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def encode_split(
    dataset,
    output_path: Path,
    encoding,
    limit: int | None,
    batch_size: int,
    num_workers: int,
) -> tuple[int, int]:
    eos = encoding.eot_token
    total_examples = len(dataset) if limit is None else min(limit, len(dataset))
    token_count = 0
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    progress = tqdm(
        total=total_examples,
        desc=f"encode {output_path.stem}",
        unit="stories",
        dynamic_ncols=True,
    )
    with temp_path.open("wb") as output:
        for start in range(0, total_examples, batch_size):
            stop = min(start + batch_size, total_examples)
            texts = dataset[start:stop]["text"]
            token_batches = encoding.encode_ordinary_batch(texts, num_threads=num_workers)
            arrays = []
            for token_ids in token_batches:
                token_ids.append(eos)
                arrays.append(np.asarray(token_ids, dtype=np.uint16))
            if arrays:
                merged = np.concatenate(arrays)
                merged.tofile(output)
                token_count += len(merged)
            progress.update(stop - start)
            progress.set_postfix(tokens=f"{token_count:,}", refresh=False)
    progress.close()
    temp_path.replace(output_path)
    return token_count, total_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/tinystories_gpt2")
    parser.add_argument("--limit-train", type=int, default=None, help="Only encode this many train stories")
    parser.add_argument("--limit-val", type=int, default=None, help="Only encode this many validation stories")
    parser.add_argument("--batch-size", type=int, default=1000, help="Stories tokenized per batch")
    parser.add_argument("--num-workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com",
    )
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    # Import after HF_ENDPOINT is configured because huggingface_hub reads it at import time.
    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("TinyStories preparation")
    print(f"  output:       {out_dir}")
    print(f"  HF endpoint:  {args.hf_endpoint or 'default'}")
    print(f"  batch size:   {args.batch_size:,} stories")
    print(f"  CPU workers:  {args.num_workers}")
    print("  tokenizer:    GPT-2 (50,257 tokens)")
    print("Downloading/loading roneneldan/TinyStories...")

    dataset = load_dataset("roneneldan/TinyStories")
    encoding = tiktoken.get_encoding("gpt2")
    train_tokens, train_stories = encode_split(
        dataset["train"],
        out_dir / "train.bin",
        encoding,
        args.limit_train,
        args.batch_size,
        args.num_workers,
    )
    val_tokens, val_stories = encode_split(
        dataset["validation"],
        out_dir / "val.bin",
        encoding,
        args.limit_val,
        args.batch_size,
        args.num_workers,
    )
    metadata = {
        "dataset": "roneneldan/TinyStories",
        "tokenizer": "gpt2",
        "vocab_size": encoding.n_vocab,
        "dtype": "uint16",
        "train_stories": train_stories,
        "val_stories": val_stories,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "hf_endpoint": args.hf_endpoint,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("TinyStories preparation complete")
    print(f"  train: {train_stories:,} stories, {train_tokens:,} tokens")
    print(f"  val:   {val_stories:,} stories, {val_tokens:,} tokens")
    print(f"  files: {out_dir / 'train.bin'} and {out_dir / 'val.bin'}")


if __name__ == "__main__":
    main()
