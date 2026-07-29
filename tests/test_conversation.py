from memory_model.conversation import (
    append_continuation_text,
    build_context_ids,
    clean_assistant_reply,
    fit_raw_context_ids,
    render_conversation,
)


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


def test_render_conversation_appends_history_and_generation_cue():
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    prompt = render_conversation(history, "你是谁？", "测试前缀")
    assert prompt == "测试前缀\n用户：你好\n助手：你好！\n用户：你是谁？\n助手："


def test_context_drops_oldest_complete_turns():
    history = [
        {"role": "user", "content": "很早的问题"},
        {"role": "assistant", "content": "很早的回答"},
        {"role": "user", "content": "较新的问题"},
        {"role": "assistant", "content": "较新的回答"},
    ]
    input_ids, dropped = build_context_ids(
        CharacterTokenizer(),
        history,
        "当前问题",
        "",
        block_size=40,
        max_new_tokens=10,
    )
    assert len(input_ids) <= 30
    assert dropped >= 2


def test_clean_reply_stops_before_next_user_turn():
    assert clean_assistant_reply("第一段回答\n用户：下一个问题") == "第一段回答"


def test_raw_continuation_does_not_inject_roles():
    context = append_continuation_text("人工智能的发展", "可以追溯到")
    assert context == "人工智能的发展\n可以追溯到"
    assert "用户：" not in context
    assert "助手：" not in context


def test_raw_context_keeps_latest_tokens():
    ids, dropped = fit_raw_context_ids(
        CharacterTokenizer(),
        "abcdefghijklmnopqrstuvwxyz",
        block_size=20,
        max_new_tokens=5,
    )
    assert len(ids) == 15
    assert dropped == 11
