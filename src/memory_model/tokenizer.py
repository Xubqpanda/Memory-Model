from __future__ import annotations


class ByteTokenizer:
    """Dependency-free tokenizer for correctness tests; one UTF-8 byte is one token."""

    vocab_size = 256
    name = "byte"
    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


class TiktokenTokenizer:
    def __init__(self, name: str = "gpt2") -> None:
        import tiktoken

        self.encoding = tiktoken.get_encoding(name)
        self.name = name
        self.vocab_size = self.encoding.n_vocab
        self.eos_token_id = self.encoding.eot_token

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special=set())

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)


def get_tokenizer(name: str):
    if name == "byte":
        return ByteTokenizer()
    return TiktokenTokenizer(name)
