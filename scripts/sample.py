#!/usr/bin/env python3
"""Generate text from a checkpoint (the context is cropped to the model's training length)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402

import torch  # noqa: E402

from cct.checkpoint import load_model  # noqa: E402
from cct.tokenizer import get_tokenizer_from_meta, load_meta  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True, help="selects the tokenizer (char vs GPT-2 BPE)")
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=1)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_meta(args.data_dir)
    tok = get_tokenizer_from_meta(meta)
    model = load_model(args.ckpt, device, vocab_size=int(meta["vocab_size"]))
    g = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    ids = tok.encode(args.prompt) or [meta.get("eot_id", 0)]
    outs = []
    for i in range(args.num_samples):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, args.max_new_tokens, args.temperature, args.top_k, args.top_p, generator=g)
        text = tok.decode(y[0].tolist())
        outs.append(text)
        if args.num_samples > 1:
            print(f"--- sample {i + 1} ---")
        print(text)
    return outs


if __name__ == "__main__":
    main()
