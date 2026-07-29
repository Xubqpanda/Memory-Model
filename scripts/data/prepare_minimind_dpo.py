#!/usr/bin/env python3
"""Encode MiniMind chosen/rejected pairs into fixed-length DPO files."""

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
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def encode_side(
    encoded_segments: list[list[int]],
    offset: int,
    segments: list[tuple[str, bool]],
    sequence_length: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
    token_ids: list[int] = []
    loss_mask: list[int] = []
    for _, supervised in segments:
        segment_ids = encoded_segments[offset]
        offset += 1
        token_ids.extend(segment_ids)
        loss_mask.extend([int(supervised)] * len(segment_ids))

    truncated = len(token_ids) > sequence_length
    token_ids = token_ids[:sequence_length]
    loss_mask = loss_mask[:sequence_length]
    real_tokens = len(token_ids)
    supervised_tokens = sum(loss_mask[1:])
    padding = sequence_length - real_tokens
    token_array = np.asarray(token_ids + [pad_token_id] * padding, dtype=np.uint16)
    mask_array = np.asarray(loss_mask + [0] * padding, dtype=np.uint8)
    return token_array, mask_array, real_tokens, supervised_tokens, truncated


def flush_records(
    records: list[tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]],
    tokenizer: HuggingFaceTokenizer,
    sequence_length: int,
    outputs: dict[tuple[str, str, str], BinaryIO],
) -> dict[str, int]:
    stats = {
        "train_examples": 0,
        "val_examples": 0,
        "chosen_tokens": 0,
        "rejected_tokens": 0,
        "chosen_supervised_tokens": 0,
        "rejected_supervised_tokens": 0,
        "chosen_truncated": 0,
        "rejected_truncated": 0,
        "skipped_without_targets": 0,
    }
    if not records:
        return stats

    pair_segments = []
    flat_texts = []
    for _, chosen, rejected in records:
        chosen_segments = render_chatml_segments(chosen, include_reasoning=False)
        rejected_segments = render_chatml_segments(rejected, include_reasoning=False)
        pair_segments.append((chosen_segments, rejected_segments))
        flat_texts.extend(text for text, _ in chosen_segments)
        flat_texts.extend(text for text, _ in rejected_segments)
    encoded_segments = tokenizer.encode_batch(flat_texts)
    offset = 0

    for (is_validation, _, _), (chosen_segments, rejected_segments) in zip(
        records, pair_segments, strict=True
    ):
        chosen_result = encode_side(
            encoded_segments,
            offset,
            chosen_segments,
            sequence_length,
            tokenizer.pad_token_id,
        )
        offset += len(chosen_segments)
        rejected_result = encode_side(
            encoded_segments,
            offset,
            rejected_segments,
            sequence_length,
            tokenizer.pad_token_id,
        )
        offset += len(rejected_segments)
        chosen_tokens, chosen_mask, chosen_real, chosen_active, chosen_truncated = chosen_result
        rejected_tokens, rejected_mask, rejected_real, rejected_active, rejected_truncated = rejected_result
        if chosen_active == 0 or rejected_active == 0:
            stats["skipped_without_targets"] += 1
            continue

        split = "val" if is_validation else "train"
        chosen_tokens.tofile(outputs[(split, "chosen", "tokens")])
        chosen_mask.tofile(outputs[(split, "chosen", "mask")])
        rejected_tokens.tofile(outputs[(split, "rejected", "tokens")])
        rejected_mask.tofile(outputs[(split, "rejected", "mask")])
        stats[f"{split}_examples"] += 1
        stats["chosen_tokens"] += chosen_real
        stats["rejected_tokens"] += rejected_real
        stats["chosen_supervised_tokens"] += chosen_active
        stats["rejected_supervised_tokens"] += rejected_active
        stats["chosen_truncated"] += int(chosen_truncated)
        stats["rejected_truncated"] += int(rejected_truncated)

    assert offset == len(encoded_segments)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/minimind/raw/dpo.jsonl")
    parser.add_argument("--out-dir", default="data/minimind/dpo")
    parser.add_argument("--tokenizer-dir", default="assets/tokenizers/minimind")
    parser.add_argument("--block-size", type=int, default=768)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.block_size <= 0 or args.batch_size <= 0:
        raise ValueError("block-size and batch-size must be positive")
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between 0 and 1")

    input_path = resolve_path(args.input)
    out_dir = resolve_path(args.out_dir)
    tokenizer = HuggingFaceTokenizer(resolve_path(args.tokenizer_dir), name="minimind")
    sequence_length = args.block_size + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    keys = [
        (split, side, kind)
        for split in ("train", "val")
        for side in ("chosen", "rejected")
        for kind in ("tokens", "mask")
    ]
    final_paths = {
        key: out_dir / f"{key[0]}_{key[1]}_{'tokens' if key[2] == 'tokens' else 'loss_mask'}.bin"
        for key in keys
    }
    temp_paths = {key: path.with_suffix(path.suffix + ".tmp") for key, path in final_paths.items()}
    totals = {
        "processed_examples": 0,
        "invalid_examples": 0,
        "prompt_mismatches": 0,
        "train_examples": 0,
        "val_examples": 0,
        "chosen_tokens": 0,
        "rejected_tokens": 0,
        "chosen_supervised_tokens": 0,
        "rejected_supervised_tokens": 0,
        "chosen_truncated": 0,
        "rejected_truncated": 0,
        "skipped_without_targets": 0,
    }

    print("MiniMind DPO data preparation")
    print(f"  input:             {input_path}")
    print(f"  output:            {out_dir}")
    print(f"  block size:        {args.block_size}")
    print(f"  validation ratio:  {args.validation_ratio:.1%}")
    print(f"  tokenizer batch:   {args.batch_size}")
    print(f"  limit:             {args.limit or 'all preference pairs'}")

    records = []
    input_size = input_path.stat().st_size
    progress = tqdm(total=input_size, desc="encode dpo", unit="B", unit_scale=True, dynamic_ncols=True)
    handles: dict[tuple[str, str, str], BinaryIO] = {}
    try:
        for key, path in temp_paths.items():
            handles[key] = path.open("wb")
        with input_path.open("rb") as source:
            for raw_line in source:
                progress.update(len(raw_line))
                if args.limit is not None and totals["processed_examples"] >= args.limit:
                    break
                try:
                    sample = json.loads(raw_line)
                    chosen = sample["chosen"]
                    rejected = sample["rejected"]
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                    totals["invalid_examples"] += 1
                    continue
                if not isinstance(chosen, list) or not isinstance(rejected, list) or not chosen or not rejected:
                    totals["invalid_examples"] += 1
                    continue
                if chosen[:-1] != rejected[:-1]:
                    totals["prompt_mismatches"] += 1
                    continue
                records.append((rng.random() < args.validation_ratio, chosen, rejected))
                totals["processed_examples"] += 1
                if len(records) >= args.batch_size:
                    batch_stats = flush_records(records, tokenizer, sequence_length, handles)
                    for key, value in batch_stats.items():
                        totals[key] += value
                    records.clear()
                    progress.set_postfix(
                        pairs=f"{totals['processed_examples']:,}",
                        train=f"{totals['train_examples']:,}",
                        val=f"{totals['val_examples']:,}",
                        refresh=False,
                    )
            batch_stats = flush_records(records, tokenizer, sequence_length, handles)
            for key, value in batch_stats.items():
                totals[key] += value
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        progress.close()
        for handle in handles.values():
            if not handle.closed:
                handle.close()

    if totals["train_examples"] == 0 or totals["val_examples"] == 0:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise RuntimeError("both train and validation splits must contain preference pairs")
    for key in keys:
        temp_paths[key].replace(final_paths[key])

    metadata = {
        "dataset": "jingyaogong/minimind_dataset",
        "training_stage": "dpo",
        "format": "chatml_chosen_rejected_assistant_only",
        "source_file": str(input_path),
        "source_size_bytes": input_size,
        "tokenizer": "minimind",
        "tokenizer_dir": str(resolve_path(args.tokenizer_dir)),
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
        **totals,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("MiniMind DPO preparation complete")
    print(f"  processed:          {totals['processed_examples']:,}")
    print(f"  train pairs:        {totals['train_examples']:,}")
    print(f"  validation pairs:   {totals['val_examples']:,}")
    print(f"  chosen truncated:   {totals['chosen_truncated']:,}")
    print(f"  rejected truncated: {totals['rejected_truncated']:,}")


if __name__ == "__main__":
    main()
