#!/usr/bin/env python3
"""Gradio chat playground for an SFT checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import gradio as gr
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_model import ModelConfig
from memory_model.conversation import build_chatml_context_ids, clean_chatml_reply
from memory_model.models import TransformerLM
from memory_model.tokenizer import get_tokenizer


class ChatHarness:
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.training_stage = checkpoint.get("training_stage")
        if self.training_stage not in {"sft", "dpo"}:
            raise ValueError("web_chat.py requires an SFT or DPO checkpoint")

        self.config = ModelConfig(**checkpoint["model_config"])
        self.model = TransformerLM(self.config).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer = get_tokenizer(checkpoint["train_config"]["tokenizer"])
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_step = int(checkpoint["step"])
        self.validation_loss = checkpoint.get("val_loss")
        self.amp_context = (
            (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
            if device.type == "cuda"
            else nullcontext
        )

    @property
    def validation_ppl(self) -> float | None:
        if self.validation_loss is None:
            return None
        return math.exp(float(self.validation_loss))

    def chat(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        greedy: bool,
        open_thinking: bool,
    ) -> tuple[list[dict[str, str]], str, str]:
        history = list(history or [])
        user_message = user_message.strip()
        if not user_message:
            return history, "", "请先输入问题。"

        input_ids, dropped_messages = build_chatml_context_ids(
            self.tokenizer,
            history,
            user_message,
            system_prompt,
            self.config.block_size,
            int(max_new_tokens),
            open_thinking=bool(open_thinking),
        )
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode(), self.amp_context():
            output = self.model.generate(
                tensor,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k) if int(top_k) > 0 else None,
                top_p=float(top_p) if float(top_p) < 1.0 else None,
                do_sample=not greedy,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0, tensor.size(1) :].tolist()
        ended_with_eos = (
            bool(generated_ids)
            and self.tokenizer.eos_token_id is not None
            and generated_ids[-1] == self.tokenizer.eos_token_id
        )
        if ended_with_eos:
            generated_ids.pop()
        decoded = self.tokenizer.decode(generated_ids)
        reply = clean_chatml_reply(decoded, open_thinking=bool(open_thinking))
        if not reply:
            reply = "（模型生成了空回复）"

        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": reply},
            ]
        )
        status = (
            f"上下文 {len(input_ids)}/{self.config.block_size} tokens · "
            f"生成 {len(generated_ids)} tokens · "
            f"{'EOS 停止' if ended_with_eos else '达到生成上限'}"
        )
        if dropped_messages:
            status += f" · 丢弃最早 {dropped_messages} 条历史消息"
        return history, "", status


def build_demo(harness: ChatHarness) -> gr.Blocks:
    description = (
        f"Stage: {harness.training_stage.upper()} · Checkpoint: {harness.checkpoint_path} · "
        f"step {harness.checkpoint_step:,}"
    )
    if harness.validation_loss is not None:
        description += f" · assistant val loss {harness.validation_loss:.4f}"
    if harness.validation_ppl is not None:
        description += f" · conditional PPL {harness.validation_ppl:.2f}"

    with gr.Blocks(title="Memory-Model Chat") as demo:
        gr.Markdown("# Memory-Model SFT 对话测试")
        gr.Markdown(
            description
            + "\n\n页面使用与 SFT 完全一致的 ChatML role 边界。PPL 只统计验证集 assistant token，"
            "不代表模型的综合问答能力。"
        )
        with gr.Row():
            with gr.Column(scale=4, min_width=560):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=560,
                    layout="bubble",
                    reasoning_tags=[("<think>", "</think>")],
                    placeholder="输入问题，测试经过 SFT 后的指令遵循和多轮对话能力。",
                )
                user_message = gr.Textbox(
                    label="你的消息",
                    lines=3,
                    placeholder="例如：请用通俗的语言解释什么是反向传播。",
                )
                with gr.Row():
                    send = gr.Button("发送", variant="primary")
                    clear = gr.Button("清空对话")
                status = gr.Markdown("等待输入。")

            with gr.Column(scale=1, min_width=320):
                gr.Markdown("## Settings")
                gr.Markdown(
                    f"上下文窗口：**{harness.config.block_size} tokens**  "
                    "\nMax tokens 会为回答预留对应的上下文空间。"
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="你是一个有帮助的中文 AI 助手。请准确、简洁地回答用户问题。",
                    lines=5,
                )
                max_new_tokens = gr.Slider(
                    16,
                    384,
                    value=192,
                    step=8,
                    label="Max tokens",
                    info="单次回答最多生成多少 token",
                )
                temperature = gr.Slider(
                    0.1,
                    1.5,
                    value=0.7,
                    step=0.05,
                    label="Temperature",
                    info="越低越稳定，越高越随机",
                )
                top_k = gr.Slider(
                    0,
                    200,
                    value=50,
                    step=1,
                    label="Top-k",
                    info="0 表示关闭",
                )
                top_p = gr.Slider(
                    0.1,
                    1.0,
                    value=0.9,
                    step=0.01,
                    label="Top-p",
                    info="核采样累计概率阈值",
                )
                greedy = gr.Checkbox(
                    value=False,
                    label="Greedy decoding",
                    info="始终选择概率最高的 token",
                )
                open_thinking = gr.Checkbox(
                    value=False,
                    label="Thinking mode",
                    info="在 Assistant 开头注入未闭合的 <think> 标签",
                )

        inputs = [
            user_message,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            greedy,
            open_thinking,
        ]
        outputs = [chatbot, user_message, status]
        send.click(harness.chat, inputs=inputs, outputs=outputs)
        user_message.submit(harness.chat, inputs=inputs, outputs=outputs)
        clear.click(
            lambda: ([], "", "对话已清空。"),
            outputs=outputs,
            queue=False,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimind_sft_60m/best.pt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    device = torch.device(args.device)
    print(f"Loading {checkpoint_path} on {device}...")
    harness = ChatHarness(checkpoint_path, device)
    demo = build_demo(harness)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
