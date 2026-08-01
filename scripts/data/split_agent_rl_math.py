#!/usr/bin/env python3
"""Create a reproducible train/validation/final-test split for RLVR data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/minimind/raw/agent_rl_math.jsonl")
    parser.add_argument("--out-dir", default="evals/data/agent_rl_math_split")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--train-size", type=int, default=18000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    out_dir = resolve_path(args.out_dir)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = args.train_size + args.val_size + args.test_size
    if len(rows) < required:
        raise ValueError(f"{input_path} has {len(rows)} rows but split requires {required}")
    if any(not isinstance(row.get("gt"), list) or not row["gt"] for row in rows):
        raise ValueError("every RLVR row must contain a non-empty gt list")

    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    selected = {
        "train": indices[: args.train_size],
        "val": indices[args.train_size : args.train_size + args.val_size],
        "test": indices[args.train_size + args.val_size : required],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_indices in selected.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as output:
            for index in split_indices:
                output.write(json.dumps(rows[index], ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "source": str(input_path),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "total_source_rows": len(rows),
        "selected_rows": required,
        "split_sizes": {key: len(value) for key, value in selected.items()},
        "source_indices": {key: value for key, value in selected.items()},
        "warning": "test.jsonl is permanently held out and must not be used for training or tuning",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("agent_rl_math split complete")
    print(f"  source rows: {len(rows):,}")
    print(f"  train:       {len(selected['train']):,}")
    print(f"  val:         {len(selected['val']):,}")
    print(f"  test:        {len(selected['test']):,}")
    print(f"  output:      {out_dir}")


if __name__ == "__main__":
    main()
