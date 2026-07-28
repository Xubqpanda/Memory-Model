#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tiny_transformer import ModelConfig, TransformerLM
from tiny_transformer.data import BinaryTokenDataset


def load_config(path: Path) -> tuple[dict, dict]:
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model, module.train


def cosine_lr(step: int, config: dict) -> float:
    if step < config["warmup_steps"]:
        return config["learning_rate"] * (step + 1) / max(1, config["warmup_steps"])
    ratio = (step - config["warmup_steps"]) / max(1, config["max_steps"] - config["warmup_steps"])
    coefficient = 0.5 * (1.0 + math.cos(math.pi * min(1.0, ratio)))
    return config["min_lr"] + coefficient * (config["learning_rate"] - config["min_lr"])


@torch.inference_mode()
def estimate_loss(model, dataset, train_config, device, amp_context) -> dict[str, float]:
    model.eval()
    result = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(train_config["eval_batches"]):
            x, y = dataset.get_batch(split, train_config["batch_size"], device)
            with amp_context():
                losses.append(model(x, targets=y).loss.item())
        result[split] = float(np.mean(losses))
    model.train()
    return result


def init_wandb(
    model_config: ModelConfig,
    train_config: dict,
    parameter_count: int,
    mode_override: str | None,
    resume_id: str | None,
):
    mode = mode_override or train_config.get("wandb_mode", "disabled")
    if mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is not installed. "
            "Install it with: python -m pip install -e '.[tracking]'"
        ) from error

    init_kwargs = dict(
        entity=train_config.get("wandb_entity"),
        project=train_config.get("wandb_project", "Memory-Model"),
        name=train_config.get("wandb_run_name"),
        tags=train_config.get("wandb_tags"),
        mode=mode,
        dir=str(PROJECT_ROOT),
        id=resume_id,
        resume="allow" if resume_id else None,
        config={
            "model": model_config.to_dict(),
            "training": train_config,
            "parameter_count": parameter_count,
        },
        settings=wandb.Settings(init_timeout=train_config.get("wandb_init_timeout", 30)),
    )
    try:
        run = wandb.init(**init_kwargs)
    except Exception as error:
        if mode != "online" or not train_config.get("wandb_fallback_offline", True):
            raise
        print(f"W&B online initialization failed ({type(error).__name__}); falling back to offline mode")
        wandb.teardown()
        init_kwargs.update(mode="offline", id=None, resume=None)
        run = wandb.init(**init_kwargs)
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("eval/*", step_metric="train/step")
    run.define_metric("performance/*", step_metric="train/step")
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_debug.py")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config for a smoke test")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Override the W&B mode from the experiment config",
    )
    parser.add_argument("--wandb-run-name", default=None, help="Override the W&B display name")
    args = parser.parse_args()

    model_dict, train_config = load_config(PROJECT_ROOT / args.config)
    train_config = dict(train_config)
    if args.max_steps is not None:
        train_config["max_steps"] = args.max_steps
    if args.wandb_run_name is not None:
        train_config["wandb_run_name"] = args.wandb_run_name

    seed = train_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)

    dtype_name = train_config["dtype"]
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_name)
    if device.type == "cuda" and amp_dtype is not None:
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    model_config = ModelConfig(**model_dict)
    model = TransformerLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=train_config["weight_decay"],
    )
    start_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1

    raw_model = model
    if train_config.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    data_dir = PROJECT_ROOT / train_config["data_dir"]
    dataset = BinaryTokenDataset(data_dir, model_config.block_size)
    out_dir = PROJECT_ROOT / train_config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    accumulation = train_config["gradient_accumulation_steps"]
    tokens_per_step = train_config["batch_size"] * model_config.block_size * accumulation
    parameter_count = raw_model.num_parameters()
    resume_id = None if checkpoint is None else checkpoint.get("wandb_run_id")
    wandb_run = init_wandb(
        model_config,
        train_config,
        parameter_count,
        args.wandb_mode,
        resume_id,
    )
    print(f"device={device}, parameters={parameter_count:,}, tokens/step={tokens_per_step:,}")

    model.train()
    for step in range(start_step, train_config["max_steps"]):
        tick = time.perf_counter()
        lr = cosine_lr(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_accumulator = 0.0

        for _ in range(accumulation):
            x, y = dataset.get_batch("train", train_config["batch_size"], device)
            with amp_context():
                loss = model(x, targets=y).loss / accumulation
            loss.backward()
            loss_accumulator += loss.detach().item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
        optimizer.step()

        if step % train_config["log_interval"] == 0:
            elapsed_ms = (time.perf_counter() - tick) * 1000
            print(f"step {step:6d} | loss {loss_accumulator:.4f} | lr {lr:.2e} | grad {grad_norm:.3f} | {elapsed_ms:.1f} ms")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step": step,
                        "train/loss": loss_accumulator,
                        "train/learning_rate": lr,
                        "train/gradient_norm": float(grad_norm),
                        "train/tokens_seen": (step + 1) * tokens_per_step,
                        "performance/step_time_ms": elapsed_ms,
                        "performance/tokens_per_second": tokens_per_step / max(elapsed_ms / 1000, 1e-9),
                    }
                )

        should_eval = step % train_config["eval_interval"] == 0 or step == train_config["max_steps"] - 1
        if should_eval:
            losses = estimate_loss(raw_model, dataset, train_config, device, amp_context)
            print(f"evaluation | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step": step,
                        "eval/train_loss": losses["train"],
                        "eval/val_loss": losses["val"],
                    }
                )
                best_val_loss = wandb_run.summary.get("best_val_loss", float("inf"))
                wandb_run.summary["best_val_loss"] = min(best_val_loss, losses["val"])
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "train_config": train_config,
                "step": step,
                "val_loss": losses["val"],
                "wandb_run_id": None if wandb_run is None else wandb_run.id,
            }
            torch.save(checkpoint, out_dir / "latest.pt")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
