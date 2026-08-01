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
from memory_model.data import PreferenceBinaryDataset
from memory_model.models import TransformerLM
from memory_model.training import (
    TrainingLogger,
    create_local_run_dir,
    disable_model_dropout,
    dpo_loss,
    initialize_distributed,
    sequence_log_probs,
)


def load_config(path: Path) -> tuple[dict, dict]:
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model, module.train


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    global_pairs_per_step = batch_size * accumulation * world_size
    steps = math.ceil(example_count / global_pairs_per_step)
    padded_count = steps * global_pairs_per_step
    generator = torch.Generator().manual_seed(seed + epoch)
    permutation = torch.randperm(example_count, generator=generator)
    if padded_count > example_count:
        needed = padded_count - example_count
        repeats = math.ceil(needed / example_count)
        permutation = torch.cat((permutation, permutation.repeat(repeats)[:needed]))
    return permutation[rank:padded_count:world_size], steps


def pair_log_probs(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    average: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat((batch["x_chosen"], batch["x_rejected"]), dim=0)
    y = torch.cat((batch["y_chosen"], batch["y_rejected"]), dim=0)
    mask = torch.cat((batch["mask_chosen"], batch["mask_rejected"]), dim=0)
    logits = model(x).logits
    logps = sequence_log_probs(logits, y, mask, average=average)
    return logps.chunk(2, dim=0)


def metric_vector(output) -> torch.Tensor:
    pair_count = output.losses.numel()
    return torch.stack(
        (
            output.losses.detach().sum(),
            output.chosen_rewards.sum(),
            output.rejected_rewards.sum(),
            output.reward_margins.sum(),
            (output.reward_margins > 0).to(torch.float32).sum(),
            output.policy_chosen_logps.sum(),
            output.policy_rejected_logps.sum(),
            output.losses.new_tensor(float(pair_count)),
        )
    )


def unpack_metrics(vector: torch.Tensor) -> dict[str, float]:
    count = max(1.0, vector[7].item())
    return {
        "loss": vector[0].item() / count,
        "chosen_reward": vector[1].item() / count,
        "rejected_reward": vector[2].item() / count,
        "reward_margin": vector[3].item() / count,
        "preference_accuracy": vector[4].item() / count,
        "policy_chosen_logp": vector[5].item() / count,
        "policy_rejected_logp": vector[6].item() / count,
    }


@torch.inference_mode()
def estimate_metrics(
    policy_model: TransformerLM,
    reference_model: TransformerLM,
    dataset: PreferenceBinaryDataset,
    train_config: dict,
    device: torch.device,
    amp_context,
    enable_tqdm: bool,
    seed: int,
) -> dict[str, float]:
    policy_model.eval()
    generator = torch.Generator().manual_seed(seed)
    total = torch.zeros(8, device=device)
    batches = tqdm(
        range(train_config["eval_batches"]),
        desc="eval dpo",
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
        batch = dataset.get_batch("val", indices, device)
        with amp_context():
            reference_chosen, reference_rejected = pair_log_probs(
                reference_model,
                batch,
                average=train_config.get("average_log_probs", False),
            )
            policy_chosen, policy_rejected = pair_log_probs(
                policy_model,
                batch,
                average=train_config.get("average_log_probs", False),
            )
            output = dpo_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta=train_config["beta"],
                label_smoothing=train_config.get("label_smoothing", 0.0),
            )
        total += metric_vector(output)
    policy_model.train()
    return unpack_metrics(total)


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

    kwargs = dict(
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
        run = wandb.init(**kwargs)
    except Exception as error:
        if mode != "online" or not train_config.get("wandb_fallback_offline", True):
            raise
        print(f"W&B online initialization failed ({type(error).__name__}); falling back to offline mode")
        wandb.teardown()
        kwargs.update(mode="offline", id=None, resume=None)
        run = wandb.init(**kwargs)
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("eval/*", step_metric="train/step")
    run.define_metric("performance/*", step_metric="train/step")
    return run


def checkpoint_payload(
    policy_model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: dict,
    step: int,
    epoch: int,
    step_in_epoch: int,
    val_metrics: dict[str, float],
    best_val_loss: float,
    wandb_run,
    local_run_dir: Path,
) -> dict:
    return {
        "model": policy_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config,
        "training_stage": "dpo",
        "step": step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "val_loss": val_metrics["loss"],
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "wandb_run_id": None if wandb_run is None else wandb_run.id,
        "local_run_dir": str(local_run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimind_dpo_60m.py")
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--reference-from", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    if args.init_from and args.resume:
        parser.error("--init-from and --resume are mutually exclusive")
    if args.compile and args.no_compile:
        parser.error("--compile and --no-compile are mutually exclusive")

    config_path = resolve_path(args.config)
    model_dict, train_config = load_config(config_path)
    train_config = dict(train_config)
    if args.init_from:
        train_config["init_from"] = args.init_from
    if args.reference_from:
        train_config["reference_from"] = args.reference_from
    if args.wandb_run_name:
        train_config["wandb_run_name"] = args.wandb_run_name
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.beta is not None:
        train_config["beta"] = args.beta
    if args.compile:
        train_config["compile"] = True
    elif args.no_compile:
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
    amp_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype))
        if device.type == "cuda" and amp_dtype is not None
        else nullcontext
    )

    model_config = ModelConfig(**model_dict)
    dataset = PreferenceBinaryDataset(resolve_path(train_config["data_dir"]))
    if dataset.block_size != model_config.block_size:
        raise ValueError("DPO data block size does not match model block size")
    accumulation = int(train_config["gradient_accumulation_steps"])
    if train_config["batch_size"] <= 0 or accumulation <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    if train_config["beta"] <= 0:
        raise ValueError("DPO beta must be positive")
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

    policy_model = TransformerLM(model_config).to(device)
    reference_model = TransformerLM(model_config).to(device)
    reference_path = resolve_path(train_config["reference_from"])
    reference_checkpoint = torch.load(reference_path, map_location="cpu", weights_only=False)
    if not model_config.matches(reference_checkpoint.get("model_config", {})):
        raise ValueError("reference checkpoint architecture does not match DPO config")
    reference_model.load_state_dict(reference_checkpoint["model"])
    del reference_checkpoint
    reference_model.requires_grad_(False)
    reference_model.eval()
    disable_model_dropout(reference_model)

    start_step = 0
    best_val_loss = float("inf")
    last_val_metrics = {"loss": float("nan")}
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(resolve_path(args.resume), map_location=device, weights_only=False)
        if checkpoint.get("training_stage") != "dpo":
            raise ValueError("--resume must point to a DPO checkpoint")
        policy_model.load_state_dict(checkpoint["model"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        last_val_metrics = dict(checkpoint.get("val_metrics", last_val_metrics))
    else:
        initial = torch.load(resolve_path(train_config["init_from"]), map_location="cpu", weights_only=False)
        if not model_config.matches(initial.get("model_config", {})):
            raise ValueError("initial policy checkpoint architecture does not match DPO config")
        policy_model.load_state_dict(initial["model"])
        del initial
    disable_model_dropout(policy_model)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=train_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=train_config["weight_decay"],
        fused=device.type == "cuda" and train_config.get("fused_optimizer", True),
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if start_step >= total_steps:
        raise ValueError(f"checkpoint has already reached total_steps={total_steps}")

    raw_policy_model = policy_model
    ddp_model = (
        DDP(
            raw_policy_model,
            device_ids=[distributed.local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        if distributed.enabled
        else raw_policy_model
    )
    policy_model = ddp_model
    if train_config.get("compile", False) and hasattr(torch, "compile"):
        policy_model = torch.compile(policy_model)

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
            description="dpo",
        )
        local_logger.save_config({"model": model_config.to_dict(), "training": train_config})
        wandb_run = init_wandb(
            model_config,
            train_config,
            raw_policy_model.num_parameters(),
            total_steps,
            args.wandb_mode,
            None if checkpoint is None else checkpoint.get("wandb_run_id"),
        )
        local_logger.write(
            f"DPO policy parameters: {raw_policy_model.num_parameters():,} | frozen reference: "
            f"{reference_model.num_parameters():,} | device={device} | world_size={distributed.world_size}"
        )
        local_logger.write(
            f"dataset: train={dataset.num_examples('train'):,}, val={dataset.num_examples('val'):,}, "
            f"block_size={dataset.block_size}"
        )
        local_logger.write(
            f"schedule: epochs={train_config['epochs']}, steps/epoch={steps_per_epoch}, "
            f"total_steps={total_steps}, per_gpu_pairs={train_config['batch_size']}, "
            f"global_pairs={global_batch_size}, beta={train_config['beta']}"
        )
        local_logger.write(
            f"policy init: {args.resume or train_config['init_from']} | reference: {reference_path} | dropout disabled"
        )

    distributed.barrier()
    policy_model.train()
    window_start = time.perf_counter()
    window_steps = 0
    window_pairs = 0

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
                    f"epoch {epoch + 1}/{train_config['epochs']} started at "
                    f"{first_epoch_step}/{steps_per_epoch}"
                )

            for epoch_step in range(first_epoch_step, steps_per_epoch):
                global_step = epoch * steps_per_epoch + epoch_step
                if global_step >= total_steps:
                    break
                lr = cosine_lr(global_step, planned_steps, train_config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                local_metrics = torch.zeros(8, device=device)
                local_start = epoch_step * accumulation * train_config["batch_size"]

                for micro_step in range(accumulation):
                    batch_start = local_start + micro_step * train_config["batch_size"]
                    indices = rank_indices[batch_start : batch_start + train_config["batch_size"]]
                    batch = dataset.get_batch("train", indices, device)
                    with torch.no_grad(), amp_context():
                        reference_chosen, reference_rejected = pair_log_probs(
                            reference_model,
                            batch,
                            average=train_config.get("average_log_probs", False),
                        )
                    sync_context = (
                        ddp_model.no_sync()
                        if distributed.enabled and micro_step < accumulation - 1
                        else nullcontext()
                    )
                    with sync_context, amp_context():
                        policy_chosen, policy_rejected = pair_log_probs(
                            policy_model,
                            batch,
                            average=train_config.get("average_log_probs", False),
                        )
                        output = dpo_loss(
                            policy_chosen,
                            policy_rejected,
                            reference_chosen,
                            reference_rejected,
                            beta=train_config["beta"],
                            label_smoothing=train_config.get("label_smoothing", 0.0),
                        )
                        loss = output.loss / accumulation
                    loss.backward()
                    local_metrics += metric_vector(output)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw_policy_model.parameters(), train_config["grad_clip"]
                )
                optimizer.step()

                completed_steps = global_step + 1
                completed_epoch = epoch_step + 1 == steps_per_epoch
                is_last_step = completed_steps == total_steps
                should_log = global_step % train_config["log_interval"] == 0 or is_last_step
                if should_log:
                    metrics = unpack_metrics(distributed.sum(local_metrics.clone()))

                if distributed.is_master:
                    local_logger.advance()
                    window_steps += 1
                    window_pairs += global_batch_size
                    if should_log:
                        elapsed = time.perf_counter() - window_start
                        step_ms = elapsed * 1000 / max(1, window_steps)
                        pairs_per_second = window_pairs / max(elapsed, 1e-9)
                        grad_norm_value = float(grad_norm)
                        train_metrics = {
                            **metrics,
                            "learning_rate": lr,
                            "gradient_norm": grad_norm_value,
                            "epoch": epoch + 1,
                            "step_in_epoch": epoch_step + 1,
                            "step_time_ms": step_ms,
                            "pairs_per_second": pairs_per_second,
                        }
                        if device.type == "cuda":
                            train_metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1024**3
                            train_metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1024**3
                        local_logger.log_metrics("train", global_step, train_metrics)
                        local_logger.write(
                            f"epoch {epoch + 1}/{train_config['epochs']} | step {epoch_step + 1:4d}/{steps_per_epoch} | "
                            f"global {global_step:5d} | loss {metrics['loss']:.4f} | "
                            f"margin {metrics['reward_margin']:.4f} | acc {metrics['preference_accuracy']:.1%} | "
                            f"lr {lr:.2e} | grad {grad_norm_value:.3f} | {pairs_per_second:.1f} pairs/s"
                        )
                        local_logger.set_postfix(
                            epoch=f"{epoch + 1}/{train_config['epochs']}",
                            loss=metrics["loss"],
                            val=last_val_metrics.get("loss"),
                            margin=metrics["reward_margin"],
                            acc=f"{metrics['preference_accuracy']:.0%}",
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "train/step": global_step,
                                    **{f"train/{key}": value for key, value in train_metrics.items()},
                                }
                            )
                        window_start = time.perf_counter()
                        window_steps = 0
                        window_pairs = 0

                should_eval = (
                    global_step % train_config["eval_interval"] == 0
                    or completed_epoch
                    or is_last_step
                )
                if should_eval:
                    distributed.barrier()
                    if distributed.is_master:
                        eval_start = time.perf_counter()
                        local_logger.write(f"DPO evaluation started at global step {global_step}")
                        last_val_metrics = estimate_metrics(
                            raw_policy_model,
                            reference_model,
                            dataset,
                            train_config,
                            device,
                            amp_context,
                            enable_tqdm=not args.no_tqdm,
                            seed=seed + global_step,
                        )
                        duration = time.perf_counter() - eval_start
                        improved = last_val_metrics["loss"] < best_val_loss
                        best_val_loss = min(best_val_loss, last_val_metrics["loss"])
                        local_logger.log_metrics(
                            "eval",
                            global_step,
                            {
                                **last_val_metrics,
                                "best_val_loss": best_val_loss,
                                "duration_seconds": duration,
                                "improved": improved,
                            },
                        )
                        local_logger.write(
                            f"evaluation step {global_step:5d} | loss {last_val_metrics['loss']:.4f} | "
                            f"margin {last_val_metrics['reward_margin']:.4f} | "
                            f"acc {last_val_metrics['preference_accuracy']:.1%} | "
                            f"best {best_val_loss:.4f} | {duration:.1f}s | improved={improved}"
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {"train/step": global_step, **{f"eval/{k}": v for k, v in last_val_metrics.items()}}
                            )
                            wandb_run.summary["best_val_loss"] = best_val_loss

                        payload = checkpoint_payload(
                            raw_policy_model,
                            optimizer,
                            model_config,
                            train_config,
                            global_step,
                            epoch,
                            epoch_step + 1,
                            last_val_metrics,
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
