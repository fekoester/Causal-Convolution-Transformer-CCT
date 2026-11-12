#!/usr/bin/env python3
"""
Ablation: does the causal-conv branch help? Trains the same small model on tiny Shakespeare with
different conv settings and an identical budget, then tabulates validation loss (bits per character).

    python -m scripts.benchmark --max-steps 3000            # ~minutes per run on a GPU; slower on CPU
    python -m scripts.benchmark --seeds 0 1 2 --max-steps 5000

Variants: plain transformer (--dyn-layers 0), CCT as in v0.1, CCT with a ReZero-gated conv branch,
two stacked convs, and a wider conv window.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

from scripts import train  # noqa: E402

VARIANTS = {
    "transformer (no conv)": ["--dyn-layers", "0"],
    "CCT v0.1 (window 2)": [],
    "CCT, ReZero conv branch": ["--conv-scale-init", "0"],
    "CCT, 2 stacked convs": ["--dyn-layers", "2", "--conv-mode", "stacked"],
    "CCT, window 8": ["--dyn-window", "8"],
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/tiny")
    p.add_argument("--out", default="results/ablation")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--model", nargs="*", default=["--seq-length", "257", "--n-embd", "192", "--n-layer", "6",
                                                   "--n-head", "6", "--batch-size", "32", "--lr", "1e-3",
                                                   "--use-scheduler", "--warmup-frac", "0.05"],
                   help="training flags shared by all variants")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    for name in a.variants:
        for seed in a.seeds:
            key = f"{name} | seed {seed}"
            if key in results:
                continue
            tag = "ablation_" + "".join(c if c.isalnum() else "_" for c in name).strip("_") + f"_s{seed}"
            print(f"\n=== {key} ===", flush=True)
            out = train.main(["--dataset", "tiny", "--data-dir", a.data_dir, "--tag", tag, "--checkpoint-root",
                              os.path.join(a.out, "runs"), "--epochs", "1000", "--max-steps", str(a.max_steps),
                              "--eval-every", str(max(1, a.max_steps // 10)), "--eval-batches", "0",
                              "--seed", str(seed), *a.model, *VARIANTS[name]])
            results[key] = {"variant": name, "seed": seed, "best_val_loss": out["best_val"],
                            "best_val_bpc": out["best_val"] / math.log(2)}
            with open(res_path, "w") as f:
                json.dump(results, f, indent=2)

    rows = []
    for name in a.variants:
        vals = [r["best_val_bpc"] for r in results.values() if r["variant"] == name]
        if vals:
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            rows.append((name, m, sd, len(vals)))
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "val_bpc_mean", "val_bpc_std", "seeds"])
        w.writerows(rows)
    print("\n| variant | best val bits/char | seeds |\n|---|---|---|")
    for name, m, sd, n in rows:
        print(f"| {name} | {m:.4f}" + (f" ± {sd:.4f}" if n > 1 else "") + f" | {n} |")


if __name__ == "__main__":
    main()
