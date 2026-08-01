import json
from collections import Counter
from pathlib import Path

from memory_model.evaluation import check_one, evaluate_checks, repetition_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


def test_fixed_chat_eval_has_100_unique_valid_rows():
    path = PROJECT_ROOT / "evals/data/fixed_chat_eval.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100
    assert all(row["messages"][-1]["role"] == "user" for row in rows)
    assert Counter(row["category"] for row in rows) == {
        "math": 20,
        "instruction": 20,
        "knowledge": 15,
        "concept": 10,
        "summary": 10,
        "multi_turn": 10,
        "repetition": 10,
        "role_eos": 5,
    }


def test_deterministic_checks_cover_numbers_json_and_constraints():
    passed, details = evaluate_checks(
        '{"answer": 42}',
        [
            {"type": "json_keys", "values": ["answer"]},
            {"type": "number", "value": "42"},
            {"type": "not_contains_any", "values": ["错误"]},
        ],
    )
    assert passed is True
    assert details == [True, True, True]
    assert check_one("1,2,3,5", {"type": "exact_compact", "value": "1, 2, 3, 5"})
    assert not check_one("144除以12等于4", {"type": "number", "value": "12"})
    assert check_one("计算过程包含12，最终答案是45。", {"type": "number", "value": "45"})


def test_repetition_metrics_detect_repeated_bigrams():
    repeated = repetition_metrics(CharacterTokenizer(), "测试测试测试")
    diverse = repetition_metrics(CharacterTokenizer(), "天地玄黄宇宙")
    assert repeated["repeat_2"] > diverse["repeat_2"]
