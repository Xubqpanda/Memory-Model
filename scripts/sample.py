#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tiny_transformer import ModelConfig, TransformerLM
from tiny_transformer.tokenizer import get_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TransformerLM(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = get_tokenizer(checkpoint["train_config"]["tokenizer"])
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        use_cache=not args.no_cache,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
