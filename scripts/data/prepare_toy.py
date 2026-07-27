#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


STORIES = """Once upon a time, a little cat lived near a blue river. The cat liked to watch the birds and play in the warm sun. One day, the cat found a red ball. It rolled the ball home and shared it with a kind dog. They played together and became good friends. The end.

Lily had a small garden. Every morning she gave water to the flowers. A yellow flower grew taller than all the others. Lily smiled and showed it to her mother. The end.

Tom saw a tiny star in the night sky. He made a wish to help his friend. The next day he carried his friend's heavy bag to school. They were both happy. The end.

"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/toy")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.frombuffer((STORIES * args.repeats).encode("utf-8"), dtype=np.uint8).astype(np.uint16)
    split = int(0.9 * len(tokens))
    tokens[:split].tofile(out_dir / "train.bin")
    tokens[split:].tofile(out_dir / "val.bin")
    metadata = {"tokenizer": "byte", "vocab_size": 256, "train_tokens": split, "val_tokens": len(tokens) - split}
    (out_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata)


if __name__ == "__main__":
    main()
