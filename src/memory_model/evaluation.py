from __future__ import annotations

import json
import re
from typing import Any


ROLE_MARKERS = ("<|im_start|>user", "<|im_start|>assistant", "用户：", "助手：")


def normalize_text(text: str) -> str:
    text = text.strip()
    return re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()


def extract_json(text: str) -> Any | None:
    text = normalize_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def number_present(text: str, value: str) -> bool:
    compact = normalize_text(text).replace(",", "")
    target = value.replace(",", "")
    # Explanatory answers often repeat numbers from the question. For an
    # objective numeric task, treat the final standalone number as the answer
    # rather than accepting any number appearing in the rationale.
    numbers = re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])", compact)
    if not numbers:
        return False
    return numbers[-1] == target


def check_one(response: str, check: dict[str, Any]) -> bool:
    kind = check["type"]
    text = normalize_text(response)
    if kind == "exact":
        return text == str(check["value"])
    if kind == "exact_casefold":
        return text.casefold() == str(check["value"]).casefold()
    if kind == "exact_compact":
        return re.sub(r"\s+", "", text) == re.sub(r"\s+", "", str(check["value"]))
    if kind == "number":
        return number_present(text, str(check["value"]))
    if kind == "contains_all":
        return all(str(value) in text for value in check["values"])
    if kind == "contains_any":
        return any(str(value) in text for value in check["values"])
    if kind == "not_contains_any":
        return all(str(value) not in text for value in check["values"])
    if kind == "starts_with":
        return text.startswith(str(check["value"]))
    if kind == "max_chars":
        return len(text) <= int(check["value"])
    if kind == "line_count":
        return len(text.splitlines()) == int(check["value"])
    if kind == "json_keys":
        value = extract_json(text)
        return isinstance(value, dict) and all(key in value for key in check["values"])
    raise ValueError(f"unknown check type: {kind}")


def evaluate_checks(response: str, checks: list[dict[str, Any]]) -> tuple[bool | None, list[bool]]:
    if not checks:
        return None, []
    results = [check_one(response, check) for check in checks]
    return all(results), results


def repetition_metrics(tokenizer, response: str) -> dict[str, float | int]:
    token_ids = tokenizer.encode(response)
    result: dict[str, float | int] = {"token_count": len(token_ids)}
    for n in (1, 2, 3):
        ngrams = [tuple(token_ids[i : i + n]) for i in range(max(0, len(token_ids) - n + 1))]
        result[f"distinct_{n}"] = len(set(ngrams)) / max(1, len(ngrams))
        result[f"repeat_{n}"] = 1.0 - float(result[f"distinct_{n}"])
    sentences = [line.strip() for line in response.splitlines() if line.strip()]
    result["repeated_line_count"] = len(sentences) - len(set(sentences))
    result["char_count"] = len(response.strip())
    return result
