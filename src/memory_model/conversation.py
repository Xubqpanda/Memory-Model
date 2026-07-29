from __future__ import annotations

import json
from typing import Any


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _render_tools_system_message(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "").strip()
    tools = _json_value(message.get("tools"))
    tool_lines = []
    if isinstance(tools, list):
        tool_lines = [json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools]

    prefix = f"{content}\n\n" if content else ""
    return (
        prefix
        + "# Tools\n\n"
        + "You may call one or more functions to assist with the user query.\n\n"
        + "You are provided with function signatures within <tools></tools> XML tags:\n"
        + "<tools>"
        + "".join(f"\n{line}" for line in tool_lines)
        + "\n</tools>\n\n"
        + "For each function call, return a json object with function name and arguments "
        + "within <tool_call></tool_call> XML tags:\n"
        + "<tool_call>\n"
        + '{"name": <function-name>, "arguments": <args-json-object>}\n'
        + "</tool_call>"
    )


def _render_tool_calls(value: Any) -> str:
    calls = _json_value(value)
    if not isinstance(calls, list):
        return ""

    rendered = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function", call)
        if not isinstance(function, dict) or not function.get("name"):
            continue
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        payload = f'{{"name": "{function["name"]}", "arguments": {arguments}}}'
        rendered.append(f"<tool_call>\n{payload}\n</tool_call>")
    return "\n".join(rendered)


def render_chatml_segments(
    conversations: list[dict[str, Any]],
    *,
    include_reasoning: bool = True,
    include_empty_think: bool = False,
) -> list[tuple[str, bool]]:
    """Render MiniMind/Qwen-style ChatML as ``(text, supervised)`` segments.

    Only assistant bodies and their closing ``<|im_end|>`` markers are
    supervised. Role headers and all context messages remain visible to the
    model but receive no direct language-model loss.
    """

    segments: list[tuple[str, bool]] = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")

        if role == "assistant":
            segments.append((f"{IM_START}assistant\n", False))
            body_parts = []
            reasoning = str(message.get("reasoning_content") or "").strip("\n")
            if include_reasoning and reasoning:
                body_parts.append(f"<think>\n{reasoning}\n</think>\n\n")
            elif include_empty_think:
                body_parts.append("<think>\n\n</think>\n\n")
            if content:
                body_parts.append(content.lstrip("\n"))
            tool_calls = _render_tool_calls(message.get("tool_calls"))
            if tool_calls:
                if body_parts and not body_parts[-1].endswith("\n"):
                    body_parts.append("\n")
                body_parts.append(tool_calls)
            body_parts.append(f"{IM_END}\n")
            segments.append(("".join(body_parts), True))
            continue

        if role == "tool":
            text = (
                f"{IM_START}user\n<tool_response>\n"
                f"{content}\n</tool_response>{IM_END}\n"
            )
            segments.append((text, False))
            continue

        if role in {"system", "user"}:
            if role == "system" and message.get("tools"):
                content = _render_tools_system_message(message)
            segments.append((f"{IM_START}{role}\n{content}{IM_END}\n", False))

    return segments


def encode_chatml_supervision(
    tokenizer: Any,
    conversations: list[dict[str, Any]],
    *,
    include_reasoning: bool = True,
    include_empty_think: bool = False,
) -> tuple[list[int], list[int]]:
    """Tokenize ChatML and return token-aligned assistant loss indicators."""

    token_ids: list[int] = []
    loss_mask: list[int] = []
    for text, supervised in render_chatml_segments(
        conversations,
        include_reasoning=include_reasoning,
        include_empty_think=include_empty_think,
    ):
        segment_ids = tokenizer.encode(text)
        token_ids.extend(segment_ids)
        loss_mask.extend([int(supervised)] * len(segment_ids))
    return token_ids, loss_mask


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


def append_continuation_text(context: str, new_text: str) -> str:
    """Append user-provided raw text without injecting role or system labels."""
    context = context.rstrip()
    new_text = new_text.strip()
    if not context:
        return new_text
    if not new_text:
        return context
    return f"{context}\n{new_text}"


def fit_raw_context_ids(
    tokenizer: Any,
    text: str,
    block_size: int,
    max_new_tokens: int,
) -> tuple[list[int], int]:
    """Keep the newest raw tokens while reserving room for continuation."""
    if not 0 < max_new_tokens < block_size:
        raise ValueError("max_new_tokens must be between 1 and block_size - 1")
    prompt_budget = block_size - max_new_tokens
    input_ids = tokenizer.encode(text)
    dropped_tokens = max(0, len(input_ids) - prompt_budget)
    return input_ids[-prompt_budget:], dropped_tokens
