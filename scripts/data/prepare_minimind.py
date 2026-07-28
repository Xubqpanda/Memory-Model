#!/usr/bin/env python3
"""Encode MiniMind pretraining JSONL into uint16 train/validation token files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_model.tokenizer import HuggingFaceTokenizer


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def flush_batch(
    records: list[tuple[bool, str]],
    tokenizer: HuggingFaceTokenizer,
    eos_token_id: int,
    train_output,
    val_output,
) -> tuple[int, int]:
    if not records:
        return 0, 0

    encoded = tokenizer.encode_batch([text for _, text in records])
    train_arrays = []
    val_arrays = []
    for (is_validation, _), token_ids in zip(records, encoded, strict=True):
        array = np.asarray([*token_ids, eos_token_id], dtype=np.uint16)
        (val_arrays if is_validation else train_arrays).append(array)

    train_tokens = 0
    if train_arrays:
        merged = np.concatenate(train_arrays)
        merged.tofile(train_output)
        train_tokens = len(merged)

    val_tokens = 0
    if val_arrays:
        merged = np.concatenate(val_arrays)
        merged.tofile(val_output)
        val_tokens = len(merged)

    return train_tokens, val_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/minimind/raw/pretrain_t2t_mini.jsonl",
        help="MiniMind JSONL file containing a text field",
    )
    parser.add_argument("--out-dir", default="data/minimind/pretrain_mini")
    parser.add_argument("--tokenizer-dir", default="assets/tokenizers/minimind")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Encode only this many documents")
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between 0 and 1")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    input_path = resolve_path(args.input)
    out_dir = resolve_path(args.out_dir)
    tokenizer_dir = resolve_path(args.tokenizer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = HuggingFaceTokenizer(tokenizer_dir, name="minimind")
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("tokenizer vocabulary does not fit in uint16")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_temp_path = train_path.with_suffix(".bin.tmp")
    val_temp_path = val_path.with_suffix(".bin.tmp")
    rng = random.Random(args.seed)

    train_examples = 0
    val_examples = 0
    train_tokens = 0
    val_tokens = 0
    processed_examples = 0
    invalid_examples = 0
    records: list[tuple[bool, str]] = []
    input_size = input_path.stat().st_size

    print("MiniMind pretraining data preparation")
    print(f"  input:             {input_path}")
    print(f"  output:            {out_dir}")
    print(f"  tokenizer:         {tokenizer_dir}")
    print(f"  vocabulary:        {tokenizer.vocab_size:,}")
    print(f"  EOS token:         {tokenizer.eos_token_id}")
    print(f"  validation ratio:  {args.validation_ratio:.2%}")
    print(f"  batch size:        {args.batch_size:,} documents")
    print(f"  limit:             {args.limit or 'all documents'}")

    progress = tqdm(
        total=input_size,
        desc="encode minimind",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    )
    with (
        input_path.open("rb") as source,
        train_temp_path.open("wb") as train_output,
        val_temp_path.open("wb") as val_output,
    ):
        for line_number, raw_line in enumerate(source, start=1):
            progress.update(len(raw_line))
            if args.limit is not None and processed_examples >= args.limit:
                break
            try:
                sample = json.loads(raw_line)
                text = sample[args.text_key]
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as error:
                invalid_examples += 1
                progress.write(f"skip invalid line {line_number}: {type(error).__name__}")
                continue
            if not isinstance(text, str) or not text.strip():
                invalid_examples += 1
                continue

            is_validation = rng.random() < args.validation_ratio
            records.append((is_validation, text))
            processed_examples += 1
            if is_validation:
                val_examples += 1
            else:
                train_examples += 1

            if len(records) >= args.batch_size:
                batch_train_tokens, batch_val_tokens = flush_batch(
                    records,
                    tokenizer,
                    tokenizer.eos_token_id,
                    train_output,
                    val_output,
                )
                train_tokens += batch_train_tokens
                val_tokens += batch_val_tokens
                records.clear()
                progress.set_postfix(
                    docs=f"{processed_examples:,}",
                    train_tokens=f"{train_tokens:,}",
                    val_tokens=f"{val_tokens:,}",
                    refresh=False,
                )

        batch_train_tokens, batch_val_tokens = flush_batch(
            records,
            tokenizer,
            tokenizer.eos_token_id,
            train_output,
            val_output,
        )
        train_tokens += batch_train_tokens
        val_tokens += batch_val_tokens
    progress.close()

    if train_tokens == 0 or val_tokens == 0:
        raise RuntimeError("both train and validation splits must contain tokens")
    train_temp_path.replace(train_path)
    val_temp_path.replace(val_path)

    metadata = {
        "dataset": "jingyaogong/minimind_dataset",
        "source_file": str(input_path),
        "source_size_bytes": input_size,
        "tokenizer": "minimind",
        "tokenizer_dir": str(tokenizer_dir),
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "dtype": "uint16",
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "processed_examples": processed_examples,
        "invalid_examples": invalid_examples,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("MiniMind preparation complete")
    print(f"  documents: {processed_examples:,} ({invalid_examples:,} invalid skipped)")
    print(f"  train:     {train_examples:,} documents, {train_tokens:,} tokens")
    print(f"  val:       {val_examples:,} documents, {val_tokens:,} tokens")
    print(f"  files:     {train_path} and {val_path}")


if __name__ == "__main__":
    main()
