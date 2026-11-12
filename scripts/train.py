#!/usr/bin/env python3
"""
Train a GPT-2 style LM with causal convolution residuals (CCT).

- CPU/GPU auto (AMP on CUDA: bf16 by default, fp16 + GradScaler with --amp-dtype fp16)
- Datasets: tiny (char), text (any local file, char), openwebtext (GPT-2 BPE)
- Validation during training (--eval-every), best.pt, CSV logs (train_log.csv, val_log.csv)
- Exact resume: model, optimizer, scaler, LR schedule, RNG and position inside the epoch;
  --epochs-add N continues for N more epochs with a fresh LR cycle over them
- --dyn-layers 0 trains the plain-transformer baseline (no conv branch)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run without install

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from contextlib import nullcontext  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.amp import GradScaler  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402

from cct import CausalConvTransformerLM, CCTConfig, LMDataset, WarmHoldCosineLR, build_optimizer  # noqa: E402
from cct.checkpoint import (  # noqa: E402,E501
    checkpoint_config,
    load_checkpoint,
    resolve_resume_arg,
    save_checkpoint,
    write_sidecar_config,
)
from cct.download_data import prepare_dataset  # noqa: E402


class EpochSampler(Sampler):
    """Deterministic shuffle per (seed, epoch), starting at `start` (for exact mid-epoch resume)."""

    def __init__(self, n: int, seed: int, epoch: int, start: int = 0):
        self.n, self.seed, self.epoch, self.start = n, seed, epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # dataset / io
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--dataset", type=str, choices=["openwebtext", "tiny", "text"], default="openwebtext")
    p.add_argument("--subset", type=str, default=None, help="HF slice for openwebtext, e.g. 'train[:1%%]'")
    p.add_argument("--max-docs", type=int, default=None, help="openwebtext: stop after this many documents")
    p.add_argument("--text-file", type=str, default=None, help="--dataset text: the text file")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--checkpoint-root", type=str, default="checkpoints")

    # model (fresh runs; on resume the checkpoint's config is used)
    p.add_argument("--seq-length", type=int, default=1024)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--dyn-window", type=int, default=2)
    p.add_argument("--dyn-layers", type=int, default=1, help="conv paths per block (0: plain transformer)")
    p.add_argument("--conv-mode", type=str, choices=["parallel", "stacked"], default="parallel")
    p.add_argument("--conv-scale-init", type=float, default=None,
                   help="learnable conv-branch scale with this initial value (0 = ReZero); default: fixed 1/K")
    p.add_argument("--dropout", type=float, default=0.0)

    # optimization
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--decay-all", action="store_true", help="v0.1: weight decay on all parameters")
    p.add_argument("--warmup-frac", type=float, default=0.01)
    p.add_argument("--hold-frac", type=float, default=0.01)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--use-scheduler", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer updates (0: off)")

    # evaluation / logging
    p.add_argument("--eval-every", type=int, default=0, help="validation every N updates (0: end of epoch only)")
    p.add_argument("--eval-batches", type=int, default=100, help="validation batches per evaluation (0: all)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=0, help="also write last.pt every N updates (0: off)")

    # hardware
    p.add_argument("--amp-dtype", type=str, choices=["bf16", "fp16"], default="bf16", help="CUDA only")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA recommended)")
    p.add_argument("--num-workers", type=int, default=2)

    # reproducibility / resume
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint file, run directory or tag under --checkpoint-root (latest checkpoint)")
    p.add_argument("--epochs-add", type=int, default=None,
                   help="train this many epochs beyond the checkpoint's epoch, with a fresh LR cycle over them")
    p.add_argument("--resume-exact", action="store_true", help="--resume must be an exact file path")
    return p


@torch.no_grad()
def evaluate(model, data_dir, seq_length, batch_size, device, amp_ctx, max_batches=0):
    """Mean token cross-entropy over the first `max_batches` validation windows (0: all)."""
    ds = LMDataset(os.path.join(data_dir, "val.bin"), seq_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with amp_ctx:
            _, loss = model(x, y)
        tot += loss.item() * y.numel()
        n += y.numel()
    if was_training:
        model.train()
    return tot / max(1, n)


class CSVLog:
    def __init__(self, path, header):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(header)

    def row(self, *vals):
        self.w.writerow(vals)
        self.f.flush()

    def close(self):
        self.f.close()


def main(argv=None):
    args = build_parser().parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    meta = prepare_dataset(args.data_dir, dataset=args.dataset, subset=args.subset, text_file=args.text_file,
                           max_docs=args.max_docs)
    vocab_size = int(meta["vocab_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
    accum = max(1, args.accum_steps)

    ckpt_path = None
    if args.resume:
        if args.resume_exact:
            if not os.path.isfile(args.resume):
                raise FileNotFoundError(f"--resume-exact given but not a file: {args.resume}")
            ckpt_path = os.path.abspath(args.resume)
        else:
            ckpt_path = resolve_resume_arg(args.resume, args.checkpoint_root)

    if ckpt_path:
        cfg_dict = checkpoint_config(ckpt_path)  # the checkpoint's own config is authoritative
        if int(cfg_dict["vocab_size"]) != vocab_size:
            raise ValueError(f"checkpoint vocab {cfg_dict['vocab_size']} != dataset vocab {vocab_size}")
    else:
        cfg_dict = dict(vocab_size=vocab_size, seq_length=args.seq_length, n_embd=args.n_embd,
                        n_layer=args.n_layer, n_head=args.n_head, dyn_window=args.dyn_window,
                        dyn_layers=args.dyn_layers, dropout=args.dropout, conv_mode=args.conv_mode,
                        conv_scale_init=args.conv_scale_init)
    config = CCTConfig.from_dict(cfg_dict)
    model = CausalConvTransformerLM(config).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay, decay_all=args.decay_all)
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    fwd = torch.compile(model) if args.compile else model

    train_ds = LMDataset(os.path.join(args.data_dir, "train.bin"), config.seq_length)
    batches_per_epoch = len(train_ds) // args.batch_size
    if batches_per_epoch == 0:
        raise ValueError("training set smaller than one batch")
    updates_per_epoch = math.ceil(batches_per_epoch / accum)

    ckpt_dir = os.path.join(args.checkpoint_root, args.tag)
    write_sidecar_config(ckpt_dir, config.to_dict(), overwrite=not ckpt_path)

    start_epoch, step, start_batch, best_val = 0, 0, 0, math.inf
    state = {}
    if ckpt_path:
        state = load_checkpoint(ckpt_path, model, optimizer, scaler)
        if state.get("batch_in_epoch") is not None:  # saved mid-epoch (last.pt)
            start_epoch, start_batch = int(state["epoch"]), int(state["batch_in_epoch"])
        else:  # end-of-epoch checkpoint (v0.1 format too)
            start_epoch = int(state.get("epoch", -1)) + 1
        step = int(state.get("step", 0))
        best_val = float(state.get("best_val", math.inf))

    target_epochs = start_epoch + max(0, args.epochs_add) if args.epochs_add is not None else args.epochs
    print(f"[train] device={device} amp={args.amp_dtype if use_amp else 'off'} accum={accum} "
          f"params={model.num_params():,} non-embedding / {model.num_params(False):,} total")
    if ckpt_path:
        print(f"[resume] {ckpt_path}: epoch {start_epoch}, batch {start_batch}, step {step}, target {target_epochs}")
    if start_epoch >= target_epochs:
        print(f"[train] start_epoch ({start_epoch}) >= target ({target_epochs}); nothing to do.")
        return {"step": step, "best_val": best_val}

    scheduler = None
    if args.use_scheduler:
        saved = (state or {}).get("scheduler")
        if ckpt_path and saved and args.epochs_add is None:
            # same run continued: restore the schedule exactly
            scheduler = WarmHoldCosineLR(optimizer, saved.get("total_steps", 1), args.warmup_frac, args.hold_frac,
                                         args.min_lr_ratio)
            scheduler.load_state_dict(saved)
        else:
            remaining = (target_epochs - start_epoch) * updates_per_epoch - math.ceil(start_batch / accum)
            for g in optimizer.param_groups:
                g["lr"] = args.lr
            scheduler = WarmHoldCosineLR(optimizer, remaining, args.warmup_frac, args.hold_frac, args.min_lr_ratio)
    if ckpt_path and state.get("rng"):
        rng = state["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if use_amp and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    train_log = CSVLog(os.path.join(ckpt_dir, "train_log.csv"), ["step", "epoch", "loss", "lr", "tokens", "time"])
    val_log = CSVLog(os.path.join(ckpt_dir, "val_log.csv"), ["step", "epoch", "val_loss", "val_ppl", "time"])
    tokens_per_batch = args.batch_size * (config.seq_length - 1)
    t0 = time.time()

    def snapshot(epoch, batch_in_epoch):
        rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
               "cuda": torch.cuda.get_rng_state_all() if use_amp else None}
        return {"epoch": epoch, "step": step, "batch_in_epoch": batch_in_epoch, "best_val": best_val, "rng": rng,
                "scheduler": scheduler.state_dict() if scheduler else None}

    def run_eval(epoch):
        nonlocal best_val
        v = evaluate(fwd, args.data_dir, config.seq_length, args.batch_size, device, amp_ctx, args.eval_batches)
        val_log.row(step, epoch, f"{v:.6f}", f"{math.exp(min(v, 50)):.3f}", f"{time.time() - t0:.1f}")
        msg = f"[val] epoch {epoch} step {step}: loss {v:.4f}, ppl {math.exp(min(v, 50)):.2f}"
        if v < best_val:
            best_val = v
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optimizer, scaler, scheduler,
                            snapshot(epoch, None) | {"best_val": v}, meta=meta)
            msg += " *"
        print(msg, flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    done = False
    for epoch in range(start_epoch, target_epochs):
        first = start_batch if epoch == start_epoch else 0
        sampler = EpochSampler(len(train_ds), args.seed, epoch, start=first * args.batch_size)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                            num_workers=args.num_workers, pin_memory=use_amp)
        win_loss, win_n = 0.0, 0
        for it, (x, y) in enumerate(loader, start=first):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp_ctx:
                _, loss = fwd(x, y)
            scaler.scale(loss / accum).backward()
            win_loss += loss.item()
            win_n += 1

            if ((it + 1) % accum == 0) or (it + 1 == batches_per_epoch):
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                step += 1

                if step % args.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    mean = win_loss / max(1, win_n)  # unscaled loss, the same on CPU and GPU
                    print(f"epoch {epoch} batch {it + 1}/{batches_per_epoch} step {step}: loss {mean:.4f} lr {lr:.2e}",
                          flush=True)
                    train_log.row(step, epoch, f"{mean:.6f}", f"{lr:.6e}", step * accum * tokens_per_batch,
                                  f"{time.time() - t0:.1f}")
                    win_loss, win_n = 0.0, 0
                if args.eval_every and step % args.eval_every == 0:
                    run_eval(epoch)
                if args.save_every and step % args.save_every == 0:
                    save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, scaler, scheduler,
                                    snapshot(epoch, it + 1), meta=meta)
                if args.max_steps and step >= args.max_steps:
                    save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, scaler, scheduler,
                                    snapshot(epoch, it + 1), meta=meta)
                    done = True
                    break
        if done:
            break
        run_eval(epoch)
        end_state = snapshot(epoch, None)
        save_checkpoint(os.path.join(ckpt_dir, f"epoch{epoch + 1}.pt"), model, optimizer, scaler, scheduler,
                        end_state, meta=meta)
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, scaler, scheduler, end_state, meta=meta)

    train_log.close()
    val_log.close()
    return {"step": step, "best_val": best_val}


if __name__ == "__main__":
    main()
