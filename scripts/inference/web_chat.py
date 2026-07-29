#!/usr/bin/env python3
"""Minimal Gradio chat harness for a pretrained Memory-Model checkpoint."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import gradio as gr
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_model import ModelConfig
from memory_model.conversation import build_context_ids, clean_assistant_reply
from memory_model.models.vanilla_transformer import TransformerLM
from memory_model.tokenizer import get_tokenizer


class ChatHarness:
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = ModelConfig(**checkpoint["model_config"])
        self.model = TransformerLM(self.config).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer = get_tokenizer(checkpoint["train_config"]["tokenizer"])
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_step = checkpoint["step"]
        self.validation_loss = checkpoint.get("val_loss")
        self.amp_context = (
            (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
            if device.type == "cuda"
            else nullcontext
        )

    def respond(
        self,
        message: str,
        history: list[dict[str, str]] | None,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        greedy: bool,
    ):
        message = message.strip()
        history = list(history or [])
        if not message:
            return history, history, "", "请输入消息。"

        input_ids, dropped_messages = build_context_ids(
            self.tokenizer,
            history,
            message,
            system_prompt,
            self.config.block_size,
            int(max_new_tokens),
        )
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with self.amp_context():
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
        reply = clean_assistant_reply(self.tokenizer.decode(generated_ids))
        if not reply:
            reply = "（模型没有生成有效文本，请尝试提高温度或修改问题。）"

        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        status = (
            f"上下文 {len(input_ids)}/{self.config.block_size} tokens · "
            f"生成 {len(generated_ids)} tokens · "
            f"{'EOS 停止' if ended_with_eos else '达到生成上限'}"
        )
        if dropped_messages:
            status += f" · 已裁剪最早 {dropped_messages // 2} 轮"
        return updated_history, updated_history, "", status


def build_demo(harness: ChatHarness) -> gr.Blocks:
    description = f"Checkpoint: {harness.checkpoint_path} · step {harness.checkpoint_step:,}"
    if harness.validation_loss is not None:
        description += f" · val loss {harness.validation_loss:.4f}"

    with gr.Blocks(title="Memory-Model Chat") as demo:
        gr.Markdown("# Memory-Model 简单对话")
        gr.Markdown(
            description
            + "\n\n这是纯预训练模型的最小对话 harness：每轮回复会追加到下一轮上下文中，"
            "尚未经过 SFT，因此角色遵循和回答质量可能不稳定。"
        )
        chatbot = gr.Chatbot(
            label="对话",
            height=560,
            layout="bubble",
            placeholder="输入一条消息，测试预训练模型的续写和简单对话能力。",
        )
        history_state = gr.State([])
        with gr.Row():
            message = gr.Textbox(
                label="消息",
                placeholder="例如：请介绍一下人工智能的发展历史。",
                lines=2,
                scale=8,
            )
            send = gr.Button("发送", variant="primary", scale=1)
        with gr.Row():
            clear = gr.Button("清空上下文")
            status = gr.Markdown("等待输入。")

        with gr.Accordion("生成参数", open=False):
            system_prompt = gr.Textbox(
                label="上下文前缀",
                value="以下是用户和助手之间的一段对话。助手会尽量准确、简洁地回答用户。",
                lines=2,
            )
            max_new_tokens = gr.Slider(16, 256, value=128, step=8, label="最大生成 tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k（0 表示关闭）")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
            greedy = gr.Checkbox(value=False, label="Greedy decoding")

        inputs = [
            message,
            history_state,
            system_prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            greedy,
        ]
        outputs = [chatbot, history_state, message, status]
        send.click(harness.respond, inputs=inputs, outputs=outputs)
        message.submit(harness.respond, inputs=inputs, outputs=outputs)
        clear.click(
            lambda: ([], [], "", "会话已清空。"),
            outputs=[chatbot, history_state, message, status],
            queue=False,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/minimind_pretrain_full_60m/best.pt",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
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
