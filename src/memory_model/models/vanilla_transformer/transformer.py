from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from .block import TransformerBlock
from .embedding import PositionEmbedding, TokenEmbedding
from .lm_head import LanguageModelHead
from .norm import TransformerLayerNorm
from .types import KVCache, ModelOutput


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = TransformerLayerNorm(config)
        self.lm_head = LanguageModelHead(config)

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
        from ...generation import sample_next_token

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
