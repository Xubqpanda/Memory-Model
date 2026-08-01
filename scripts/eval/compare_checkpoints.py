#!/usr/bin/env python3
"""Compare SFT/DPO checkpoints on the permanently held-out fixed chat suite."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_model.evaluation import ROLE_MARKERS, evaluate_checks, repetition_metrics
from scripts.inference.web_chat import ChatHarness


DEFAULT_SYSTEM = "你是一个有帮助的中文 AI 助手。请准确、简洁地回答用户问题。"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate_checkpoint(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    system_prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    harness = ChatHarness(checkpoint_path, device)
    tokenizer = harness.tokenizer
    outputs = []
    category_values: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row_index, row in enumerate(tqdm(rows, desc=checkpoint_path.parent.name, unit="prompt")):
        messages = row["messages"]
        history = list(messages[:-1])
        user_message = messages[-1]["content"]
        max_new_tokens = int(row.get("max_new_tokens", 128))
        torch.manual_seed(seed + row_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + row_index)
        history_after, _, status = harness.chat(
            user_message,
            history,
            system_prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            greedy,
            False,
        )
        response = history_after[-1]["content"]
        checks_pass, check_results = evaluate_checks(response, row.get("checks", []))
        repetition = repetition_metrics(tokenizer, response)
        role_leakage = any(marker in response for marker in ROLE_MARKERS)
        eos_stopped = "EOS 停止" in status
        result = {
            "id": row["id"],
            "category": row["category"],
            "response": response,
            "status": status,
            "eos_stopped": eos_stopped,
            "role_leakage": role_leakage,
            "checks_pass": checks_pass,
            "check_results": check_results,
            **repetition,
        }
        outputs.append(result)
        category_values[row["category"]].append(result)

    def mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "samples": len(outputs),
        "eos_stop_rate": mean([float(x["eos_stopped"]) for x in outputs]),
        "role_leakage_rate": mean([float(x["role_leakage"]) for x in outputs]),
        "mean_tokens": mean([float(x["token_count"]) for x in outputs]),
        "mean_chars": mean([float(x["char_count"]) for x in outputs]),
        "mean_distinct_1": mean([float(x["distinct_1"]) for x in outputs]),
        "mean_distinct_2": mean([float(x["distinct_2"]) for x in outputs]),
        "mean_distinct_3": mean([float(x["distinct_3"]) for x in outputs]),
        "mean_repeat_2": mean([float(x["repeat_2"]) for x in outputs]),
        "mean_repeat_3": mean([float(x["repeat_3"]) for x in outputs]),
        "open_ended_samples": sum(x["checks_pass"] is None for x in outputs),
        "checked_samples": sum(x["checks_pass"] is not None for x in outputs),
        "checked_accuracy": mean(
            [float(x["checks_pass"]) for x in outputs if x["checks_pass"] is not None]
        ),
        "by_category": {},
    }
    for category, values in sorted(category_values.items()):
        summary["by_category"][category] = {
            "samples": len(values),
            "checked_accuracy": mean(
                [float(x["checks_pass"]) for x in values if x["checks_pass"] is not None]
            ),
            "eos_stop_rate": mean([float(x["eos_stopped"]) for x in values]),
            "role_leakage_rate": mean([float(x["role_leakage"]) for x in values]),
            "mean_tokens": mean([float(x["token_count"]) for x in values]),
            "mean_distinct_2": mean([float(x["distinct_2"]) for x in values]),
        }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", default="evals/data/fixed_chat_eval.jsonl")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path or label=path; pass twice to compare SFT and DPO",
    )
    parser.add_argument("--output-dir", default="evals/results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    eval_path = resolve_path(args.eval_data)
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    run_name = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_specs = []
    for value in args.checkpoint:
        if "=" in value:
            label, checkpoint = value.split("=", 1)
        else:
            checkpoint = value
            path = resolve_path(checkpoint)
            label = f"{path.parent.name}_{path.stem}"
        checkpoint_specs.append((label, checkpoint))
    if len({label for label, _ in checkpoint_specs}) != len(checkpoint_specs):
        raise ValueError("checkpoint labels must be unique")

    config = {
        "eval_data": str(eval_path),
        "checkpoints": checkpoint_specs,
        "device": str(device),
        "system_prompt": args.system_prompt,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "greedy": args.greedy,
        "seed": args.seed,
        "samples": len(rows),
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    all_summaries = []
    for label, checkpoint in checkpoint_specs:
        path = resolve_path(checkpoint)
        print(f"Evaluating {path} ({len(rows)} prompts)...", flush=True)
        outputs, summary = evaluate_checkpoint(
            path,
            rows,
            device,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=args.greedy,
            seed=args.seed,
        )
        (run_dir / f"{label}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
            encoding="utf-8",
        )
        all_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    (run_dir / "summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
