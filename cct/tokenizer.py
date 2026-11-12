from __future__ import annotations

import json
import os
from typing import Any

EOT_ID = 50256  # GPT-2 <|endoftext|>

_tokenizer = None


class _GPT2:
    """GPT-2 BPE with encode / encode_ordinary / decode: tiktoken if installed, else transformers."""

    def __init__(self):
        try:
            import tiktoken

            enc = tiktoken.get_encoding("gpt2")
            self.encode_ordinary = enc.encode_ordinary
            self.encode = enc.encode_ordinary
            self.decode = enc.decode
        except ImportError:
            from transformers import GPT2TokenizerFast

            tok = GPT2TokenizerFast.from_pretrained("gpt2")
            self.encode_ordinary = lambda s: tok.encode(s)
            self.encode = self.encode_ordinary
            self.decode = lambda ids: tok.decode(ids)


def get_gpt2_tokenizer() -> _GPT2:
    """Lazily create the GPT-2 tokenizer (modules that only need meta helpers do not import it)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _GPT2()
    return _tokenizer


class CharTokenizer:
    """Character-level tokenizer; expects meta with 'itos' (list of characters)."""

    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {ch: i for i, ch in enumerate(itos)}

    def encode(self, s: str) -> list[int]:
        unknown = sorted({ch for ch in s if ch not in self.stoi})
        if unknown:
            print(f"[cct] warning: characters not in the vocabulary are dropped: {''.join(unknown)!r}")
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "?" for i in ids)


def load_meta(data_dir: str) -> dict[str, Any]:
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"meta.json not found in {data_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_tokenizer_from_meta(meta: dict[str, Any]):
    if meta.get("tokenizer") == "char" or meta.get("dataset") in ("tiny", "text"):
        return CharTokenizer(meta["itos"])
    return get_gpt2_tokenizer()
