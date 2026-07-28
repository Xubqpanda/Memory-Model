from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


class HuggingFaceTokenizer:
    """Load a tokenizer.json asset without depending on Transformers."""

    def __init__(self, tokenizer_dir: str | Path, name: str | None = None) -> None:
        from tokenizers import Tokenizer

        tokenizer_dir = Path(tokenizer_dir)
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = PROJECT_ROOT / tokenizer_dir
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        self.name = name or tokenizer_dir.name
        self.vocab_size = self.tokenizer.get_vocab_size()

        config_path = tokenizer_dir / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.bos_token_id = self.tokenizer.token_to_id(config["bos_token"])
        self.eos_token_id = self.tokenizer.token_to_id(config["eos_token"])
        self.pad_token_id = self.tokenizer.token_to_id(config["pad_token"])

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [
            encoding.ids
            for encoding in self.tokenizer.encode_batch(texts, add_special_tokens=False)
        ]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def get_tokenizer(name: str):
    if name == "byte":
        return ByteTokenizer()
    if name == "minimind":
        return HuggingFaceTokenizer("assets/tokenizers/minimind", name="minimind")
    if name.startswith("hf:"):
        return HuggingFaceTokenizer(name.removeprefix("hf:"), name=name)
    return TiktokenTokenizer(name)
