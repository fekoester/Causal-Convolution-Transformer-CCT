"""
Prepare tokenized datasets: train.bin / val.bin (uint16) + meta.json in a data directory.

  tiny          tiny Shakespeare, character level (downloaded)
  text          any local text file, character level (--text-file)
  openwebtext   OpenWebText, GPT-2 BPE, streamed from the Hugging Face Hub (Parquet copy), documents
                separated by the end-of-text token; tokens are written to disk in chunks, so memory
                stays flat for the full corpus (~9B tokens)
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import numpy as np
from tqdm import tqdm

from .tokenizer import EOT_ID, get_gpt2_tokenizer

OWT_SOURCES = ("dylanebert/openwebtext", "Skylion007/openwebtext")
CHUNK = 1 << 22


def _save_meta(data_dir: str, meta: dict[str, Any]) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_meta_if_exists(data_dir: str) -> dict[str, Any] | None:
    path = os.path.join(data_dir, "meta.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _already_prepared(data_dir: str) -> bool:
    return all(os.path.exists(os.path.join(data_dir, f)) for f in ("train.bin", "val.bin", "meta.json"))


def _split_to_files(all_path: str, data_dir: str, val_fraction: float) -> tuple[int, int]:
    """Copy the last `val_fraction` of the tokens in all_path to val.bin, the rest to train.bin."""
    src = np.memmap(all_path, dtype=np.uint16, mode="r")
    n = len(src)
    n_train = n - max(1, int(n * val_fraction))
    for name, a, b in (("train.bin", 0, n_train), ("val.bin", n_train, n)):
        out = np.memmap(os.path.join(data_dir, name), dtype=np.uint16, mode="w+", shape=(b - a,))
        for i in range(a, b, CHUNK):
            j = min(i + CHUNK, b)
            out[i - a: j - a] = src[i:j]
        out.flush()
        del out
    del src
    os.remove(all_path)
    return n_train, n - n_train


def prepare_char_level(text: str, data_dir: str, dataset_name: str, val_fraction: float = 0.1) -> None:
    """Character-level dataset from a string: sorted character vocabulary, 90/10 split."""
    os.makedirs(data_dir, exist_ok=True)
    itos = sorted(set(text))
    if len(itos) > 65535:
        raise ValueError("character vocabulary does not fit in uint16")
    stoi = {ch: i for i, ch in enumerate(itos)}
    ids = np.fromiter((stoi[c] for c in text), dtype=np.uint16, count=len(text))
    split = int((1.0 - val_fraction) * len(ids))
    ids[:split].tofile(os.path.join(data_dir, "train.bin"))
    ids[split:].tofile(os.path.join(data_dir, "val.bin"))
    _save_meta(data_dir, {"dataset": dataset_name, "tokenizer": "char", "itos": itos, "vocab_size": len(itos)})
    print(f"[cct] {dataset_name}: vocab {len(itos)}, {split} train / {len(ids) - split} val characters")


def _prepare_tiny(data_dir: str) -> None:
    import requests

    print("[cct] Downloading tiny Shakespeare (char-level)…")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    prepare_char_level(r.text, data_dir, "tiny")


def _iter_owt(subset: str | None, max_docs: int | None):
    from datasets import load_dataset

    last_err = None
    for name in OWT_SOURCES:
        try:
            if subset:  # an HF split slice such as "train[:1%]" (downloads that part)
                ds = load_dataset(name, split=subset)
            else:
                ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception as e:  # try the next mirror
            last_err = e
    else:
        raise RuntimeError(f"Could not load OpenWebText from {OWT_SOURCES}: {last_err}")
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        yield ex["text"]


def _prepare_openwebtext(data_dir: str, subset: str | None = None, max_docs: int | None = None,
                         val_fraction: float = 0.1) -> None:
    os.makedirs(data_dir, exist_ok=True)
    tok = get_gpt2_tokenizer()
    all_path = os.path.join(data_dir, "all.tmp.bin")
    n, buf = 0, []
    print(f"[cct] Tokenizing OpenWebText ({subset or 'full, streamed'}) with GPT-2 BPE…")
    with open(all_path, "wb") as f:
        for text in tqdm(_iter_owt(subset, max_docs), desc="Tokenizing OWT", unit="doc"):
            ids = tok.encode_ordinary(text)
            if ids:
                buf.extend(ids)
                buf.append(EOT_ID)
            if len(buf) >= CHUNK:
                np.asarray(buf, dtype=np.uint16).tofile(f)
                n += len(buf)
                buf = []
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)
            n += len(buf)
    if n == 0:
        raise RuntimeError("No tokens produced. Check dataset download/tokenizer.")
    n_train, n_val = _split_to_files(all_path, data_dir, val_fraction)
    _save_meta(data_dir, {"dataset": "openwebtext", "tokenizer": "gpt2", "vocab_size": 50257, "eot_id": EOT_ID})
    print(f"[cct] OpenWebText ready: {n_train} train / {n_val} val tokens")


def prepare_dataset(
    data_dir: str,
    dataset: Literal["openwebtext", "tiny", "text"] = "openwebtext",
    subset: str | None = None,
    text_file: str | None = None,
    max_docs: int | None = None,
) -> dict[str, Any]:
    """Ensure a tokenized dataset exists in data_dir (train.bin, val.bin, meta.json); return its meta."""
    if not _already_prepared(data_dir):
        if dataset == "tiny":
            _prepare_tiny(data_dir)
        elif dataset == "text":
            if not text_file:
                raise ValueError("--dataset text needs --text-file")
            with open(text_file, encoding="utf-8") as f:
                prepare_char_level(f.read(), data_dir, "text")
        else:
            _prepare_openwebtext(data_dir, subset=subset, max_docs=max_docs)
    else:
        print(f"[cct] Found existing dataset in {data_dir}")
    meta = _load_meta_if_exists(data_dir)
    if meta is None:
        raise RuntimeError("prepare_dataset failed to write meta.json")
    return meta
