from memory_model.tokenizer import get_tokenizer


def test_minimind_tokenizer_round_trip_and_special_tokens():
    tokenizer = get_tokenizer("minimind")
    text = "你好，MiniMind！"
    token_ids = tokenizer.encode(text)

    assert tokenizer.vocab_size == 6400
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 0
    assert tokenizer.decode(token_ids) == text
