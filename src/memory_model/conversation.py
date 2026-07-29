from __future__ import annotations

from typing import Any


def render_conversation(
    history: list[dict[str, str]],
    user_message: str,
    system_prompt: str = "",
) -> str:
    """Render a minimal plain-text user/assistant transcript."""
    parts = []
    if system_prompt.strip():
        parts.append(system_prompt.strip())
    for message in history:
        role = message.get("role")
        content = message.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"用户：{content}")
        elif role == "assistant":
            parts.append(f"助手：{content}")
    parts.append(f"用户：{user_message.strip()}")
    parts.append("助手：")
    return "\n".join(parts)


def build_context_ids(
    tokenizer: Any,
    history: list[dict[str, str]],
    user_message: str,
    system_prompt: str,
    block_size: int,
    max_new_tokens: int,
) -> tuple[list[int], int]:
    """Fit a transcript into the context window by dropping oldest turns."""
    if not 0 < max_new_tokens < block_size:
        raise ValueError("max_new_tokens must be between 1 and block_size - 1")

    prompt_budget = block_size - max_new_tokens
    retained = list(history)
    dropped_messages = 0
    while True:
        prompt = render_conversation(retained, user_message, system_prompt)
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) <= prompt_budget:
            return input_ids, dropped_messages
        if retained:
            drop_count = 2 if len(retained) >= 2 else 1
            retained = retained[drop_count:]
            dropped_messages += drop_count
            continue
        return input_ids[-prompt_budget:], dropped_messages


def clean_assistant_reply(text: str) -> str:
    """Keep only the first assistant turn if the base model continues roles."""
    reply = text.strip()
    for marker in ("\n用户：", "\nUser:", "<|im_start|>user"):
        if marker in reply:
            reply = reply.split(marker, 1)[0].rstrip()
    return reply
