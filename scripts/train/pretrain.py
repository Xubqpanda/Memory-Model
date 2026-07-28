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
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_model import ModelConfig
from memory_model.data import BinaryTokenDataset
from memory_model.models.vanilla_transformer import TransformerLM
from memory_model.training import (
    TrainingLogger,
    create_local_run_dir,
    initialize_distributed,
    resolve_gradient_accumulation_steps,
)


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
def estimate_loss(model, dataset, train_config, device, amp_context, enable_tqdm: bool) -> dict[str, float]:
    model.eval()
    result = {}
    for split in ("train", "val"):
        losses = []
        batches = tqdm(
            range(train_config["eval_batches"]),
            desc=f"eval {split}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not enable_tqdm,
        )
        for _ in batches:
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


def checkpoint_payload(
    raw_model,
    optimizer,
    model_config,
    train_config,
    step: int,
    val_loss: float,
    best_val_loss: float,
    wandb_run,
    local_run_dir: Path,
) -> dict:
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config,
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "wandb_run_id": None if wandb_run is None else wandb_run.id,
        "local_run_dir": str(local_run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_debug.py")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config for a smoke test")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable the terminal progress bar")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Override the W&B mode from the experiment config",
    )
    parser.add_argument("--wandb-run-name", default=None, help="Override the W&B display name")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    model_dict, train_config = load_config(config_path)
    train_config = dict(train_config)
    if args.max_steps is not None:
        train_config["max_steps"] = args.max_steps
    if args.wandb_run_name is not None:
        train_config["wandb_run_name"] = args.wandb_run_name

    distributed = initialize_distributed(args.device)
    device = distributed.device

    # All ranks initialize the same model; DDP then guarantees identical parameters.
    seed = train_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        fused=device.type == "cuda" and train_config.get("fused_optimizer", True),
    )

    start_step = 0
    checkpoint = None
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1

    raw_model = model
    distributed_model = (
        DDP(
            raw_model,
            device_ids=[distributed.local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        if distributed.enabled
        else raw_model
    )
    model = distributed_model
    if train_config.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Different ranks must sample different training windows and dropout masks.
    rank_seed = seed + distributed.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)

    data_dir = PROJECT_ROOT / train_config["data_dir"]
    dataset = BinaryTokenDataset(data_dir, model_config.block_size)
    out_dir = PROJECT_ROOT / train_config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    accumulation = resolve_gradient_accumulation_steps(
        batch_size=train_config["batch_size"],
        block_size=model_config.block_size,
        world_size=distributed.world_size,
        configured_steps=train_config["gradient_accumulation_steps"],
        target_tokens_per_step=train_config.get("target_tokens_per_step"),
    )
    train_config["resolved_gradient_accumulation_steps"] = accumulation
    train_config["world_size"] = distributed.world_size
    tokens_per_step = (
        train_config["batch_size"]
        * model_config.block_size
        * accumulation
        * distributed.world_size
    )
    configured_epoch_steps = list(train_config.get("epoch_checkpoint_steps", []))
    if configured_epoch_steps != sorted(set(configured_epoch_steps)):
        raise ValueError("epoch_checkpoint_steps must be unique and strictly increasing")
    if any(step <= 0 for step in configured_epoch_steps):
        raise ValueError("epoch_checkpoint_steps must contain positive completed-step counts")
    epoch_checkpoint_by_step = {
        completed_step: epoch_number
        for epoch_number, completed_step in enumerate(configured_epoch_steps, start=1)
        if completed_step <= train_config["max_steps"]
    }
    parameter_count = raw_model.num_parameters()
    total_planned_tokens = train_config["max_steps"] * tokens_per_step
    remaining_tokens = max(0, train_config["max_steps"] - start_step) * tokens_per_step
    equivalent_epochs = total_planned_tokens / len(dataset.train)

    run_name = train_config.get("wandb_run_name") or config_path.stem
    resume_dir = None if checkpoint is None else checkpoint.get("local_run_dir")
    local_run_dir = create_local_run_dir(PROJECT_ROOT, run_name, resume_dir)
    local_logger = None
    wandb_run = None
    if distributed.is_master:
        local_logger = TrainingLogger(
            local_run_dir,
            total_steps=train_config["max_steps"],
            start_step=start_step,
            enable_tqdm=not args.no_tqdm,
        )
        local_logger.save_config(
            {
                "config_path": str(config_path),
                "model": model_config.to_dict(),
                "training": train_config,
                "runtime": {
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "world_size": distributed.world_size,
                    "parameter_count": parameter_count,
                    "tokens_per_step": tokens_per_step,
                    "total_planned_tokens": total_planned_tokens,
                    "remaining_tokens": remaining_tokens,
                    "train_dataset_tokens": len(dataset.train),
                    "val_dataset_tokens": len(dataset.val),
                    "equivalent_train_epochs": equivalent_epochs,
                    "start_step": start_step,
                },
            }
        )

        resume_id = None if checkpoint is None else checkpoint.get("wandb_run_id")
        wandb_run = init_wandb(
            model_config,
            train_config,
            parameter_count,
            args.wandb_mode,
            resume_id,
        )

        local_logger.write(f"run directory: {local_run_dir}")
        local_logger.write(f"checkpoint directory: {out_dir}")
        local_logger.write(
            f"device={device} | world_size={distributed.world_size} | dtype={dtype_name} | "
            f"parameters={parameter_count:,} | layers={model_config.n_layer} | "
            f"heads={model_config.n_head} | d_model={model_config.d_model}"
        )
        local_logger.write(
            f"train tokens={len(dataset.train):,} | val tokens={len(dataset.val):,} | "
            f"global tokens/step={tokens_per_step:,} | planned tokens={total_planned_tokens:,} | "
            f"equivalent epochs={equivalent_epochs:.2f}"
        )
        local_logger.write(
            f"steps={start_step:,}->{train_config['max_steps']:,} | per-GPU batch={train_config['batch_size']} | "
            f"sequence={model_config.block_size} | accumulation={accumulation} | "
            f"compile={train_config.get('compile', False)} | "
            f"fused optimizer={device.type == 'cuda' and train_config.get('fused_optimizer', True)}"
        )
        if epoch_checkpoint_by_step:
            local_logger.write(
                "epoch checkpoints: "
                + ", ".join(
                    f"epoch {epoch} at completed step {completed_step:,}"
                    for completed_step, epoch in epoch_checkpoint_by_step.items()
                )
            )

    # Workers wait while rank 0 initializes W&B and local logs.
    distributed.barrier()

    best_val_loss = float("inf") if checkpoint is None else checkpoint.get("best_val_loss", float("inf"))
    last_val_loss = None if checkpoint is None else checkpoint.get("val_loss")
    window_start = time.perf_counter()
    window_steps = 0
    window_tokens = 0

    model.train()
    try:
        for step in range(start_step, train_config["max_steps"]):
            lr = cosine_lr(step, train_config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_accumulator = 0.0

            for micro_step in range(accumulation):
                x, y = dataset.get_batch("train", train_config["batch_size"], device)
                sync_context = (
                    distributed_model.no_sync()
                    if distributed.enabled and micro_step < accumulation - 1
                    else nullcontext()
                )
                with sync_context:
                    with amp_context():
                        loss = model(x, targets=y).loss / accumulation
                    loss.backward()
                loss_accumulator += loss.detach().item()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if distributed.is_master:
                local_logger.advance()
                window_steps += 1
                window_tokens += tokens_per_step

            is_last_step = step == train_config["max_steps"] - 1
            completed_steps = step + 1
            completed_epoch = epoch_checkpoint_by_step.get(completed_steps)
            should_log = step % train_config["log_interval"] == 0 or is_last_step
            mean_loss = loss_accumulator
            if should_log:
                mean_loss = distributed.mean(torch.tensor(loss_accumulator, device=device)).item()
                if distributed.is_master:
                    elapsed = time.perf_counter() - window_start
                    average_step_ms = elapsed * 1000 / max(window_steps, 1)
                    tokens_per_second = window_tokens / max(elapsed, 1e-9)
                    grad_norm_value = float(grad_norm)
                    train_metrics = {
                        "loss": mean_loss,
                        "learning_rate": lr,
                        "gradient_norm": grad_norm_value,
                        "tokens_seen": (step + 1) * tokens_per_step,
                        "step_time_ms": average_step_ms,
                        "tokens_per_second": tokens_per_second,
                    }
                    if device.type == "cuda":
                        train_metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1024**3
                        train_metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1024**3

                    local_logger.log_metrics("train", step, train_metrics)
                    local_logger.write(
                        f"step {step:6d}/{train_config['max_steps'] - 1} | "
                        f"loss {mean_loss:.4f} | lr {lr:.3e} | grad {grad_norm_value:.3f} | "
                        f"{tokens_per_second:,.0f} tok/s | {average_step_ms:.1f} ms/step"
                    )
                    local_logger.set_postfix(
                        loss=mean_loss,
                        val=last_val_loss,
                        lr=f"{lr:.2e}",
                        grad=grad_norm_value,
                        tok_s=f"{tokens_per_second:,.0f}",
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/step": step,
                                "train/loss": mean_loss,
                                "train/learning_rate": lr,
                                "train/gradient_norm": grad_norm_value,
                                "train/tokens_seen": (step + 1) * tokens_per_step,
                                "performance/step_time_ms": average_step_ms,
                                "performance/tokens_per_second": tokens_per_second,
                                **{
                                    f"performance/{key}": value
                                    for key, value in train_metrics.items()
                                    if key.startswith("gpu_memory_")
                                },
                            }
                        )
                    window_start = time.perf_counter()
                    window_steps = 0
                    window_tokens = 0

            should_eval = (
                step % train_config["eval_interval"] == 0
                or is_last_step
                or completed_epoch is not None
            )
            if should_eval:
                distributed.barrier()
                if distributed.is_master:
                    eval_start = time.perf_counter()
                    local_logger.write(f"evaluation started at step {step}")
                    losses = estimate_loss(
                        raw_model,
                        dataset,
                        train_config,
                        device,
                        amp_context,
                        enable_tqdm=not args.no_tqdm,
                    )
                    eval_seconds = time.perf_counter() - eval_start
                    last_val_loss = losses["val"]
                    improved = losses["val"] < best_val_loss
                    best_val_loss = min(best_val_loss, losses["val"])
                    eval_metrics = {
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "best_val_loss": best_val_loss,
                        "duration_seconds": eval_seconds,
                        "improved": improved,
                    }
                    local_logger.log_metrics("eval", step, eval_metrics)
                    local_logger.write(
                        f"evaluation step {step:6d} | train {losses['train']:.4f} | "
                        f"val {losses['val']:.4f} | best {best_val_loss:.4f} | "
                        f"duration {eval_seconds:.1f}s | improved={improved}"
                    )
                    local_logger.set_postfix(loss=mean_loss, val=losses["val"], lr=f"{lr:.2e}")

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/step": step,
                                "eval/train_loss": losses["train"],
                                "eval/val_loss": losses["val"],
                                "eval/duration_seconds": eval_seconds,
                            }
                        )
                        wandb_run.summary["best_val_loss"] = best_val_loss

                    payload = checkpoint_payload(
                        raw_model,
                        optimizer,
                        model_config,
                        train_config,
                        step,
                        losses["val"],
                        best_val_loss,
                        wandb_run,
                        local_run_dir,
                    )
                    if completed_epoch is not None:
                        payload["completed_epoch"] = completed_epoch
                    latest_path = out_dir / "latest.pt"
                    torch.save(payload, latest_path)
                    if improved:
                        torch.save(payload, out_dir / "best.pt")
                    epoch_path = None
                    if completed_epoch is not None:
                        epoch_path = out_dir / f"epoch_{completed_epoch}.pt"
                        torch.save(payload, epoch_path)
                    local_logger.write(
                        f"checkpoint saved: {latest_path}"
                        + (f" and {out_dir / 'best.pt'}" if improved else "")
                        + (f" and {epoch_path}" if epoch_path is not None else "")
                    )
                    # Do not charge evaluation/checkpoint time to the next training throughput window.
                    window_start = time.perf_counter()
                distributed.barrier()
    finally:
        if distributed.is_master:
            if wandb_run is not None:
                wandb_run.finish()
            local_logger.close()
        distributed.close()


if __name__ == "__main__":
    main()
