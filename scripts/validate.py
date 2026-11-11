#!/usr/bin/env python3
"""Validation loss and perplexity of a checkpoint on the data directory's val.bin."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import math  # noqa: E402
from contextlib import nullcontext  # noqa: E402

import torch  # noqa: E402

from cct.checkpoint import load_model  # noqa: E402
from cct.tokenizer import load_meta  # noqa: E402
from scripts.train import evaluate  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=0, help="0: the whole validation set")
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_meta(args.data_dir)
    model = load_model(args.ckpt, device, vocab_size=int(meta["vocab_size"]))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    loss = evaluate(model, args.data_dir, model.config.seq_length, args.batch_size, device, amp, args.max_batches)
    print(f"Validation loss {loss:.4f}, perplexity {math.exp(min(loss, 50)):.2f}"
          + (f", {loss / math.log(2):.4f} bits per token" if meta.get("tokenizer") == "char" else ""))
    return loss


if __name__ == "__main__":
    main()
