#!/usr/bin/env python3
"""Plot train / validation loss from the CSV logs that scripts.train writes into each run directory."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="*", help="run directories (default: every run under --log-dir)")
    p.add_argument("--log-dir", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default="loss.png")
    p.add_argument("--x", choices=["step", "tokens"], default="step")
    args = p.parse_args(argv)

    runs = args.runs or sorted(os.path.join(args.log_dir, d) for d in os.listdir(args.log_dir)
                               if os.path.isdir(os.path.join(args.log_dir, d)))
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in runs:
        name = os.path.basename(os.path.normpath(run))
        tr, va = os.path.join(run, "train_log.csv"), os.path.join(run, "val_log.csv")
        color = None
        if os.path.exists(tr):
            df = pd.read_csv(tr)
            (line,) = ax.plot(df[args.x] if args.x in df else df["step"], df["loss"], alpha=0.5, lw=1,
                              label=f"{name} train")
            color = line.get_color()
        if os.path.exists(va) and args.x == "step":
            df = pd.read_csv(va)
            ax.plot(df["step"], df["val_loss"], marker="o", ms=3, color=color, label=f"{name} val")
    ax.set_xlabel(args.x)
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
