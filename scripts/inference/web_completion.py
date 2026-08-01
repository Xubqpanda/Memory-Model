#!/usr/bin/env python3
"""Raw-text Gradio completion playground for a pretrained checkpoint."""

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
from memory_model.conversation import append_continuation_text, fit_raw_context_ids
from memory_model.models import TransformerLM
from memory_model.tokenizer import get_tokenizer


class CompletionHarness:
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

    def continue_text(
        self,
        added_text: str,
        context: str | None,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        greedy: bool,
    ):
        context = context or ""
        raw_prompt = append_continuation_text(context, added_text)
        if not raw_prompt:
            return context, context, "", "请先输入一段开头。"

        input_ids, dropped_tokens = fit_raw_context_ids(
            self.tokenizer,
            raw_prompt,
            self.config.block_size,
            int(max_new_tokens),
        )
        retained_prompt = self.tokenizer.decode(input_ids)
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
        continuation = self.tokenizer.decode(generated_ids)
        updated_context = retained_prompt + continuation
        status = (
            f"输入上下文 {len(input_ids)}/{self.config.block_size} tokens · "
            f"续写 {len(generated_ids)} tokens · "
            f"{'EOS 停止' if ended_with_eos else '达到生成上限'}"
        )
        if dropped_tokens:
            status += f" · 已从最前面裁剪 {dropped_tokens} tokens"
        return updated_context, updated_context, "", status


def build_demo(harness: CompletionHarness) -> gr.Blocks:
    description = f"Checkpoint: {harness.checkpoint_path} · step {harness.checkpoint_step:,}"
    if harness.validation_loss is not None:
        description += f" · val loss {harness.validation_loss:.4f}"

    with gr.Blocks(title="Memory-Model Completion") as demo:
        gr.Markdown("# Memory-Model 预训练续写测试")
        gr.Markdown(
            description
            + "\n\n页面不会注入 user、assistant 或 system role。模型只看到下方累计的原始文本，"
            "每次生成结果都会原样追加到上下文。"
        )
        context_display = gr.Textbox(
            label="累计原文与模型续写",
            lines=24,
            interactive=False,
            placeholder="累计文本会显示在这里。",
        )
        context_state = gr.State("")
        added_text = gr.Textbox(
            label="追加文本",
            placeholder="输入一个开头，例如：人工智能的发展历史可以追溯到",
            lines=4,
        )
        with gr.Row():
            generate = gr.Button("追加并续写", variant="primary")
            continue_button = gr.Button("继续续写（无需追加文本）")
            clear = gr.Button("清空")
        status = gr.Markdown("等待输入。")

        with gr.Accordion("生成参数", open=False):
            max_new_tokens = gr.Slider(16, 256, value=128, step=8, label="最大生成 tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k（0 表示关闭）")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
            greedy = gr.Checkbox(value=False, label="Greedy decoding")

        common_inputs = [
            context_state,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            greedy,
        ]
        outputs = [context_display, context_state, added_text, status]
        generate.click(
            harness.continue_text,
            inputs=[added_text, *common_inputs],
            outputs=outputs,
        )
        added_text.submit(
            harness.continue_text,
            inputs=[added_text, *common_inputs],
            outputs=outputs,
        )
        continue_button.click(
            harness.continue_text,
            inputs=[gr.State(""), *common_inputs],
            outputs=outputs,
        )
        clear.click(
            lambda: ("", "", "", "上下文已清空。"),
            outputs=outputs,
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
    harness = CompletionHarness(checkpoint_path, device)
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
