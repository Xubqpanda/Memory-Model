from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[KVCache] | None = None


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, including an inference KV cache."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch_size, seq_len, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # A query at absolute position p may only see keys at positions <= p.
        # For ordinary training, is_causal=True lets PyTorch use its fastest kernel.
        if past_len == 0:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            query_positions = past_len + torch.arange(seq_len, device=x.device)
            key_positions = torch.arange(k.size(2), device=x.device)
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, width)
        y = self.resid_dropout(self.out_proj(y))
        return y, (k, v) if use_cache else None


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    """Pre-Norm block: x + Attention(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, new_cache = self.attn(self.attn_norm(x), past_key_value, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-style scaled residual initialization.
        for name, parameter in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        _, seq_len = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if past_len + seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {past_len + seq_len} exceeds block_size={self.config.block_size}"
            )
        positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)

        new_caches: list[KVCache] | None = [] if use_cache else None
        for layer_idx, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[layer_idx]
            x, layer_cache = block(x, layer_past, use_cache)
            if new_caches is not None:
                assert layer_cache is not None
                new_caches.append(layer_cache)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return ModelOutput(logits=logits, loss=loss, past_key_values=new_caches)

    def num_parameters(self, non_embedding: bool = False) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            count -= self.position_embedding.weight.numel()
        return count

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> torch.Tensor:
        from .generation import sample_next_token

        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        was_training = self.training
        self.eval()
        generated = input_ids
        cache = None

        for _ in range(max_new_tokens):
            # Learned absolute positions cannot exceed block_size. If the context
            # becomes too long, use the latest window and rebuild the cache.
            if use_cache and cache is not None and generated.size(1) <= self.config.block_size:
                model_input = generated[:, -1:]
            else:
                model_input = generated[:, -self.config.block_size :]
                cache = None

            output = self(model_input, past_key_values=cache, use_cache=use_cache)
            cache = output.past_key_values
            next_token = sample_next_token(
                output.logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            generated = torch.cat((generated, next_token), dim=1)

        self.train(was_training)
        return generated
