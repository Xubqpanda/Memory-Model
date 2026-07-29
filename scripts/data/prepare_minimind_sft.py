#!/usr/bin/env python3
"""Encode MiniMind conversations into fixed-length SFT token/mask files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_model.conversation import render_chatml_segments
from memory_model.tokenizer import HuggingFaceTokenizer


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def encode_records(
    records: list[tuple[bool, list[dict[str, Any]], bool]],
    tokenizer: HuggingFaceTokenizer,
    sequence_length: int,
    pad_token_id: int,
    train_tokens_file: BinaryIO,
    train_masks_file: BinaryIO,
    val_tokens_file: BinaryIO,
    val_masks_file: BinaryIO,
    include_reasoning: bool,
) -> dict[str, int]:
    if not records:
        return {"train_examples": 0, "val_examples": 0, "train_tokens": 0, "val_tokens": 0}

    document_segments: list[list[tuple[str, bool]]] = []
    flat_texts: list[str] = []
    for _, conversations, include_empty_think in records:
        segments = render_chatml_segments(
            conversations,
            include_reasoning=include_reasoning,
            include_empty_think=include_empty_think,
        )
        document_segments.append(segments)
        flat_texts.extend(text for text, _ in segments)

    encoded_segments = tokenizer.encode_batch(flat_texts)
    encoded_offset = 0
    stats = {"train_examples": 0, "val_examples": 0, "train_tokens": 0, "val_tokens": 0}

    for (is_validation, _, _), segments in zip(records, document_segments, strict=True):
        token_ids: list[int] = []
        loss_mask: list[int] = []
        for _, supervised in segments:
            segment_ids = encoded_segments[encoded_offset]
            encoded_offset += 1
            token_ids.extend(segment_ids)
            loss_mask.extend([int(supervised)] * len(segment_ids))

        token_ids = token_ids[:sequence_length]
        loss_mask = loss_mask[:sequence_length]
        if not any(loss_mask[1:]):
            continue

        real_tokens = len(token_ids)
        padding = sequence_length - real_tokens
        token_array = np.asarray(token_ids + [pad_token_id] * padding, dtype=np.uint16)
        mask_array = np.asarray(loss_mask + [0] * padding, dtype=np.uint8)

        if is_validation:
            token_array.tofile(val_tokens_file)
            mask_array.tofile(val_masks_file)
            stats["val_examples"] += 1
            stats["val_tokens"] += real_tokens
        else:
            token_array.tofile(train_tokens_file)
            mask_array.tofile(train_masks_file)
            stats["train_examples"] += 1
            stats["train_tokens"] += real_tokens

    assert encoded_offset == len(encoded_segments)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/minimind/raw/sft_t2t_mini.jsonl")
    parser.add_argument("--out-dir", default="data/minimind/sft_mini")
    parser.add_argument("--tokenizer-dir", default="assets/tokenizers/minimind")
    parser.add_argument(
        "--block-size",
        type=int,
        default=768,
        help="Number of model input positions; encoded rows contain block_size + 1 tokens",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=512, help="Conversations per tokenizer batch")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--exclude-reasoning",
        action="store_true",
        help="Do not place reasoning_content inside assistant responses",
    )
    parser.add_argument(
        "--empty-think-ratio",
        type=float,
        default=0.2,
        help="Probability of retaining empty <think> tags for non-reasoning answers",
    )
    args = parser.parse_args()

    if args.block_size <= 0:
        raise ValueError("block-size must be positive")
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between 0 and 1")
    if not 0.0 <= args.empty_think_ratio <= 1.0:
        raise ValueError("empty-think-ratio must be between 0 and 1")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    input_path = resolve_path(args.input)
    out_dir = resolve_path(args.out_dir)
    tokenizer_dir = resolve_path(args.tokenizer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = HuggingFaceTokenizer(tokenizer_dir, name="minimind")
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("tokenizer vocabulary does not fit in uint16")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define a pad token")

    sequence_length = args.block_size + 1
    output_names = (
        "train_tokens.bin",
        "train_loss_mask.bin",
        "val_tokens.bin",
        "val_loss_mask.bin",
    )
    final_paths = [out_dir / name for name in output_names]
    temp_paths = [path.with_suffix(path.suffix + ".tmp") for path in final_paths]
    rng = random.Random(args.seed)
    input_size = input_path.stat().st_size

    totals = {
        "processed_examples": 0,
        "invalid_examples": 0,
        "skipped_without_assistant_targets": 0,
        "train_examples": 0,
        "val_examples": 0,
        "train_tokens": 0,
        "val_tokens": 0,
    }

    print("MiniMind SFT data preparation")
    print(f"  input:               {input_path}")
    print(f"  output:              {out_dir}")
    print(f"  tokenizer:           {tokenizer_dir}")
    print(f"  vocabulary:          {tokenizer.vocab_size:,}")
    print(f"  block size:          {args.block_size:,}")
    print(f"  encoded row length:  {sequence_length:,}")
    print(f"  validation ratio:    {args.validation_ratio:.2%}")
    print(f"  include reasoning:   {not args.exclude_reasoning}")
    print(f"  empty think ratio:   {args.empty_think_ratio:.0%}")
    print(f"  tokenizer batch:     {args.batch_size:,} conversations")
    print(f"  limit:               {args.limit or 'all conversations'}")

    records: list[tuple[bool, list[dict[str, Any]], bool]] = []

    def flush() -> None:
        before = len(records)
        batch_stats = encode_records(
            records,
            tokenizer,
            sequence_length,
            tokenizer.pad_token_id,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            include_reasoning=not args.exclude_reasoning,
        )
        kept = batch_stats["train_examples"] + batch_stats["val_examples"]
        totals["skipped_without_assistant_targets"] += before - kept
        for key, value in batch_stats.items():
            totals[key] += value
        records.clear()

    progress = tqdm(total=input_size, desc="encode sft", unit="B", unit_scale=True, dynamic_ncols=True)
    try:
        with (
            input_path.open("rb") as source,
            temp_paths[0].open("wb") as train_tokens_file,
            temp_paths[1].open("wb") as train_masks_file,
            temp_paths[2].open("wb") as val_tokens_file,
            temp_paths[3].open("wb") as val_masks_file,
        ):
            outputs = (train_tokens_file, train_masks_file, val_tokens_file, val_masks_file)
            for line_number, raw_line in enumerate(source, start=1):
                progress.update(len(raw_line))
                if args.limit is not None and totals["processed_examples"] >= args.limit:
                    break
                try:
                    sample = json.loads(raw_line)
                    conversations = sample["conversations"]
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                    totals["invalid_examples"] += 1
                    continue
                if not isinstance(conversations, list) or not conversations:
                    totals["invalid_examples"] += 1
                    continue

                records.append(
                    (
                        rng.random() < args.validation_ratio,
                        conversations,
                        rng.random() < args.empty_think_ratio,
                    )
                )
                totals["processed_examples"] += 1
                if len(records) >= args.batch_size:
                    flush()
                    progress.set_postfix(
                        examples=f"{totals['processed_examples']:,}",
                        train=f"{totals['train_examples']:,}",
                        val=f"{totals['val_examples']:,}",
                        refresh=False,
                    )
            flush()
    except BaseException:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        progress.close()

    if totals["train_examples"] == 0 or totals["val_examples"] == 0:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        raise RuntimeError("both train and validation splits must contain supervised examples")

    for temp_path, final_path in zip(temp_paths, final_paths, strict=True):
        temp_path.replace(final_path)

    metadata = {
        "dataset": "jingyaogong/minimind_dataset",
        "training_stage": "sft",
        "format": "chatml_assistant_only",
        "source_file": str(input_path),
        "source_size_bytes": input_size,
        "tokenizer": "minimind",
        "tokenizer_dir": str(tokenizer_dir),
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "token_dtype": "uint16",
        "loss_mask_dtype": "uint8",
        "block_size": args.block_size,
        "sequence_length": sequence_length,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "include_reasoning": not args.exclude_reasoning,
        "empty_think_ratio": args.empty_think_ratio,
        **totals,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("MiniMind SFT preparation complete")
    print(f"  parsed:       {totals['processed_examples']:,} conversations")
    print(f"  invalid:      {totals['invalid_examples']:,}")
    print(f"  no targets:   {totals['skipped_without_assistant_targets']:,}")
    print(f"  train:        {totals['train_examples']:,} examples")
    print(f"  validation:   {totals['val_examples']:,} examples")
    print(f"  output files: {out_dir}")


if __name__ == "__main__":
    main()
