from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def mean(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
        return tensor

    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(device_arg: str) -> DistributedContext:
    enabled = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return DistributedContext(True, rank, local_rank, world_size, torch.device("cuda", local_rank))

    return DistributedContext(False, 0, 0, 1, torch.device(device_arg))


def resolve_gradient_accumulation_steps(
    batch_size: int,
    block_size: int,
    world_size: int,
    configured_steps: int,
    target_tokens_per_step: int | None,
) -> int:
    if target_tokens_per_step is None:
        return configured_steps

    tokens_per_micro_step = batch_size * block_size * world_size
    if target_tokens_per_step < tokens_per_micro_step:
        raise ValueError(
            f"target_tokens_per_step={target_tokens_per_step:,} is smaller than one distributed "
            f"micro-step ({tokens_per_micro_step:,} tokens)"
        )
    if target_tokens_per_step % tokens_per_micro_step != 0:
        raise ValueError(
            f"target_tokens_per_step={target_tokens_per_step:,} must be divisible by "
            f"batch_size × block_size × world_size = {tokens_per_micro_step:,}"
        )
    return target_tokens_per_step // tokens_per_micro_step
