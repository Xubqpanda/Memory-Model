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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_model import ModelConfig
from memory_model.data import SupervisedBinaryDataset
from memory_model.models.vanilla_transformer import TransformerLM
from memory_model.training import TrainingLogger, create_local_run_dir, initialize_distributed


def load_config(path: Path) -> tuple[dict, dict]:
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model, module.train


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def cosine_lr(step: int, total_steps: int, config: dict) -> float:
    if step < config["warmup_steps"]:
        return config["learning_rate"] * (step + 1) / max(1, config["warmup_steps"])
    ratio = (step - config["warmup_steps"]) / max(1, total_steps - config["warmup_steps"])
    coefficient = 0.5 * (1.0 + math.cos(math.pi * min(1.0, ratio)))
    return config["min_lr"] + coefficient * (config["learning_rate"] - config["min_lr"])


def epoch_rank_indices(
    example_count: int,
    batch_size: int,
    accumulation: int,
    world_size: int,
    rank: int,
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, int]:
    global_examples_per_step = batch_size * accumulation * world_size
    steps = math.ceil(example_count / global_examples_per_step)
    padded_count = steps * global_examples_per_step
    generator = torch.Generator().manual_seed(seed + epoch)
    permutation = torch.randperm(example_count, generator=generator)
    if padded_count > example_count:
        needed = padded_count - example_count
        repeats = math.ceil(needed / example_count)
        permutation = torch.cat((permutation, permutation.repeat(repeats)[:needed]))
    return permutation[rank:padded_count:world_size], steps


@torch.inference_mode()
def estimate_loss(
    model: TransformerLM,
    dataset: SupervisedBinaryDataset,
    train_config: dict,
    device: torch.device,
    amp_context,
    enable_tqdm: bool,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    loss_sum = 0.0
    supervised_tokens = 0
    batches = tqdm(
        range(train_config["eval_batches"]),
        desc="eval sft",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not enable_tqdm,
    )
    for _ in batches:
        indices = torch.randint(
            dataset.num_examples("val"),
            (train_config["batch_size"],),
            generator=generator,
        )
        x, y, active = dataset.get_batch("val", indices, device)
        with amp_context():
            logits = model(x).logits
            batch_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        loss_sum += batch_loss.item()
        supervised_tokens += active
    model.train()
    return loss_sum / max(1, supervised_tokens)


def init_wandb(
    model_config: ModelConfig,
    train_config: dict,
    parameter_count: int,
    total_steps: int,
    mode_override: str | None,
    resume_id: str | None,
):
    mode = mode_override or train_config.get("wandb_mode", "disabled")
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install W&B with: python -m pip install -e '.[tracking]'") from error

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
            "total_steps": total_steps,
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
    raw_model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: dict,
    step: int,
    epoch: int,
    step_in_epoch: int,
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
        "training_stage": "sft",
        "step": step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "wandb_run_id": None if wandb_run is None else wandb_run.id,
        "local_run_dir": str(local_run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimind_sft_60m.py")
    parser.add_argument("--init-from", default=None, help="Override pretrained checkpoint")
    parser.add_argument("--resume", default=None, help="Resume an SFT checkpoint including optimizer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap optimizer steps for a smoke test")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    if args.init_from and args.resume:
        parser.error("--init-from and --resume are mutually exclusive")

    config_path = resolve_path(args.config)
    model_dict, train_config = load_config(config_path)
    train_config = dict(train_config)
    if args.init_from:
        train_config["init_from"] = args.init_from
    if args.wandb_run_name:
        train_config["wandb_run_name"] = args.wandb_run_name
    if args.no_compile:
        train_config["compile"] = False

    distributed = initialize_distributed(args.device)
    device = distributed.device
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
    dataset = SupervisedBinaryDataset(resolve_path(train_config["data_dir"]))
    if dataset.block_size != model_config.block_size:
        raise ValueError(
            f"SFT data block_size={dataset.block_size} does not match model block_size={model_config.block_size}"
        )

    accumulation = int(train_config["gradient_accumulation_steps"])
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    global_batch_size = train_config["batch_size"] * accumulation * distributed.world_size
    steps_per_epoch = math.ceil(dataset.num_examples("train") / global_batch_size)
    planned_steps = train_config["epochs"] * steps_per_epoch
    total_steps = min(planned_steps, args.max_steps) if args.max_steps is not None else planned_steps
    train_config.update(
        world_size=distributed.world_size,
        global_batch_size=global_batch_size,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
    )

    model = TransformerLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=train_config["weight_decay"],
        fused=device.type == "cuda" and train_config.get("fused_optimizer", True),
    )

    start_step = 0
    best_val_loss = float("inf")
    last_val_loss = float("nan")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(resolve_path(args.resume), map_location=device, weights_only=False)
        if checkpoint.get("training_stage") != "sft":
            raise ValueError("--resume must point to an SFT checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        last_val_loss = float(checkpoint.get("val_loss", last_val_loss))
    else:
        init_path = resolve_path(train_config["init_from"])
        initial = torch.load(init_path, map_location="cpu", weights_only=False)
        checkpoint_config = initial.get("model_config")
        if checkpoint_config is not None and checkpoint_config != model_config.to_dict():
            raise ValueError("pretraining checkpoint architecture does not match SFT config")
        model.load_state_dict(initial["model"])
        del initial

    if start_step >= total_steps:
        raise ValueError(f"checkpoint step {start_step} has already reached total_steps={total_steps}")

    raw_model = model
    ddp_model = (
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
    model = ddp_model
    if train_config.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    rank_seed = seed + distributed.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)

    out_dir = resolve_path(train_config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = train_config.get("wandb_run_name") or config_path.stem
    resume_dir = None if checkpoint is None else checkpoint.get("local_run_dir")
    local_run_dir = create_local_run_dir(PROJECT_ROOT, run_name, resume_dir)
    local_logger = None
    wandb_run = None

    if distributed.is_master:
        local_logger = TrainingLogger(
            local_run_dir,
            total_steps=total_steps,
            start_step=start_step,
            enable_tqdm=not args.no_tqdm,
            description="sft",
        )
        local_logger.save_config({"model": model_config.to_dict(), "training": train_config})
        wandb_run = init_wandb(
            model_config,
            train_config,
            raw_model.num_parameters(),
            total_steps,
            args.wandb_mode,
            None if checkpoint is None else checkpoint.get("wandb_run_id"),
        )
        local_logger.write(
            f"SFT model parameters: {raw_model.num_parameters():,} | device={device} | "
            f"world_size={distributed.world_size}"
        )
        local_logger.write(
            f"dataset: train={dataset.num_examples('train'):,}, val={dataset.num_examples('val'):,}, "
            f"block_size={dataset.block_size}, assistant-only loss"
        )
        local_logger.write(
            f"schedule: epochs={train_config['epochs']}, steps/epoch={steps_per_epoch:,}, "
            f"total_steps={total_steps:,}, per_gpu_batch={train_config['batch_size']}, "
            f"global_batch={global_batch_size}"
        )
        local_logger.write(
            f"initialization: {'resume ' + args.resume if args.resume else train_config['init_from']}"
        )

    distributed.barrier()
    model.train()
    window_start = time.perf_counter()
    window_steps = 0
    window_input_tokens = 0
    window_supervised_tokens = 0

    try:
        for epoch in range(start_step // steps_per_epoch, train_config["epochs"]):
            rank_indices, actual_steps = epoch_rank_indices(
                dataset.num_examples("train"),
                train_config["batch_size"],
                accumulation,
                distributed.world_size,
                distributed.rank,
                seed,
                epoch,
            )
            assert actual_steps == steps_per_epoch
            first_epoch_step = start_step % steps_per_epoch if epoch == start_step // steps_per_epoch else 0
            if distributed.is_master:
                local_logger.write(
                    f"epoch {epoch + 1}/{train_config['epochs']} started at step_in_epoch "
                    f"{first_epoch_step}/{steps_per_epoch}"
                )

            for epoch_step in range(first_epoch_step, steps_per_epoch):
                global_step = epoch * steps_per_epoch + epoch_step
                if global_step >= total_steps:
                    break
                lr = cosine_lr(global_step, planned_steps, train_config)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                local_batches = []
                local_supervised = 0
                local_input_tokens = 0
                local_start = epoch_step * accumulation * train_config["batch_size"]
                for micro_step in range(accumulation):
                    batch_start = local_start + micro_step * train_config["batch_size"]
                    batch_indices = rank_indices[batch_start : batch_start + train_config["batch_size"]]
                    x, y, active = dataset.get_batch("train", batch_indices, device)
                    local_batches.append((x, y))
                    local_supervised += active
                    local_input_tokens += x.numel()

                global_supervised_tensor = distributed.sum(
                    torch.tensor(local_supervised, dtype=torch.long, device=device)
                )
                global_supervised = int(global_supervised_tensor.item())
                if global_supervised == 0:
                    raise RuntimeError("training batch has no assistant tokens")

                optimizer.zero_grad(set_to_none=True)
                local_loss_sum = torch.zeros((), device=device)
                for micro_step, (x, y) in enumerate(local_batches):
                    sync_context = (
                        ddp_model.no_sync()
                        if distributed.enabled and micro_step < accumulation - 1
                        else nullcontext()
                    )
                    with sync_context:
                        with amp_context():
                            logits = model(x).logits
                            loss_sum = F.cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1),
                                ignore_index=-100,
                                reduction="sum",
                            )
                            # DDP averages gradients across ranks. Multiplying by
                            # world_size makes the final gradient a true global
                            # supervised-token mean rather than a mean of rank means.
                            scaled_loss = loss_sum * distributed.world_size / global_supervised
                        scaled_loss.backward()
                    local_loss_sum += loss_sum.detach()

                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_config["grad_clip"])
                optimizer.step()

                completed_steps = global_step + 1
                completed_epoch = epoch_step + 1 == steps_per_epoch
                is_last_step = completed_steps == total_steps
                should_log = global_step % train_config["log_interval"] == 0 or is_last_step
                if should_log:
                    global_loss_sum = distributed.sum(local_loss_sum.clone()).item()
                    mean_loss = global_loss_sum / global_supervised
                else:
                    mean_loss = float("nan")

                if distributed.is_master:
                    local_logger.advance()
                    window_steps += 1
                    window_input_tokens += local_input_tokens * distributed.world_size
                    window_supervised_tokens += global_supervised

                    if should_log:
                        elapsed = time.perf_counter() - window_start
                        step_ms = elapsed * 1000 / max(1, window_steps)
                        input_tokens_per_second = window_input_tokens / max(elapsed, 1e-9)
                        supervised_tokens_per_second = window_supervised_tokens / max(elapsed, 1e-9)
                        grad_norm_value = float(grad_norm)
                        metrics = {
                            "loss": mean_loss,
                            "learning_rate": lr,
                            "gradient_norm": grad_norm_value,
                            "epoch": epoch + 1,
                            "step_in_epoch": epoch_step + 1,
                            "supervised_tokens": global_supervised,
                            "step_time_ms": step_ms,
                            "input_tokens_per_second": input_tokens_per_second,
                            "supervised_tokens_per_second": supervised_tokens_per_second,
                        }
                        if device.type == "cuda":
                            metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1024**3
                            metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1024**3
                        local_logger.log_metrics("train", global_step, metrics)
                        local_logger.write(
                            f"epoch {epoch + 1}/{train_config['epochs']} | "
                            f"step {epoch_step + 1:5d}/{steps_per_epoch} | global {global_step:6d} | "
                            f"loss {mean_loss:.4f} | lr {lr:.3e} | grad {grad_norm_value:.3f} | "
                            f"{supervised_tokens_per_second:,.0f} supervised tok/s"
                        )
                        local_logger.set_postfix(
                            epoch=f"{epoch + 1}/{train_config['epochs']}",
                            loss=mean_loss,
                            val=last_val_loss,
                            lr=f"{lr:.2e}",
                            sup_tok_s=f"{supervised_tokens_per_second:,.0f}",
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "train/step": global_step,
                                    **{f"train/{key}": value for key, value in metrics.items() if not key.startswith("gpu_")},
                                    **{f"performance/{key}": value for key, value in metrics.items() if key.startswith("gpu_") or key.endswith("per_second") or key == "step_time_ms"},
                                }
                            )
                        window_start = time.perf_counter()
                        window_steps = 0
                        window_input_tokens = 0
                        window_supervised_tokens = 0

                should_eval = (
                    global_step % train_config["eval_interval"] == 0
                    or completed_epoch
                    or is_last_step
                )
                if should_eval:
                    distributed.barrier()
                    if distributed.is_master:
                        eval_start = time.perf_counter()
                        local_logger.write(f"SFT evaluation started at global step {global_step}")
                        last_val_loss = estimate_loss(
                            raw_model,
                            dataset,
                            train_config,
                            device,
                            amp_context,
                            enable_tqdm=not args.no_tqdm,
                            seed=seed + global_step,
                        )
                        duration = time.perf_counter() - eval_start
                        improved = last_val_loss < best_val_loss
                        best_val_loss = min(best_val_loss, last_val_loss)
                        local_logger.log_metrics(
                            "eval",
                            global_step,
                            {
                                "val_loss": last_val_loss,
                                "best_val_loss": best_val_loss,
                                "duration_seconds": duration,
                                "improved": improved,
                            },
                        )
                        local_logger.write(
                            f"evaluation step {global_step:6d} | val {last_val_loss:.4f} | "
                            f"best {best_val_loss:.4f} | duration {duration:.1f}s | improved={improved}"
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "train/step": global_step,
                                    "eval/val_loss": last_val_loss,
                                    "eval/duration_seconds": duration,
                                }
                            )
                            wandb_run.summary["best_val_loss"] = best_val_loss

                        payload = checkpoint_payload(
                            raw_model,
                            optimizer,
                            model_config,
                            train_config,
                            global_step,
                            epoch,
                            epoch_step + 1,
                            last_val_loss,
                            best_val_loss,
                            wandb_run,
                            local_run_dir,
                        )
                        latest_path = out_dir / "latest.pt"
                        torch.save(payload, latest_path)
                        saved = [str(latest_path)]
                        if improved:
                            best_path = out_dir / "best.pt"
                            torch.save(payload, best_path)
                            saved.append(str(best_path))
                        if completed_epoch:
                            epoch_path = out_dir / f"epoch_{epoch + 1}.pt"
                            torch.save(payload, epoch_path)
                            saved.append(str(epoch_path))
                        local_logger.write("checkpoint saved: " + ", ".join(saved))
                        window_start = time.perf_counter()
                    distributed.barrier()

            if (epoch + 1) * steps_per_epoch >= total_steps:
                break
    finally:
        if distributed.is_master:
            if wandb_run is not None:
                wandb_run.finish()
            local_logger.close()
        distributed.close()


if __name__ == "__main__":
    main()
