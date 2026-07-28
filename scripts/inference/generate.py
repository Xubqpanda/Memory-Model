#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_model import ModelConfig
from memory_model.models.vanilla_transformer import TransformerLM
from memory_model.tokenizer import get_tokenizer


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
    parser.add_argument("--ignore-eos", action="store_true", help="Continue until max-new-tokens")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        eos_token_id=None if args.ignore_eos else tokenizer.eos_token_id,
    )
    token_ids = output[0].tolist()
    if tokenizer.eos_token_id is not None and token_ids[-1] == tokenizer.eos_token_id:
        token_ids.pop()
    print(tokenizer.decode(token_ids))


if __name__ == "__main__":
    main()
