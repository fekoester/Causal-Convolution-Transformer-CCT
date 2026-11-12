import json
import os
import re
from typing import Any

import torch

from .model import CausalConvTransformerLM, CCTConfig


def save_checkpoint(path, model, optimizer, scaler, scheduler, state: dict, config: dict | None = None,
                    meta: dict | None = None):
    """Model, optimizer, AMP scaler, scheduler, training state (epoch, step, RNG) and the model config."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler": scaler.state_dict() if scaler else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "state": state,
        "config": config if config is not None else model.config.to_dict(),
        "meta": meta,
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint behind


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None) -> dict[str, Any]:
    """Load weights (and optionally optimizer / scaler / scheduler state); returns the training state."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model"].items()}
    getattr(model, "_orig_mod", model).load_state_dict(sd)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    sched_state = ckpt.get("scheduler") or (ckpt.get("state") or {}).get("scheduler")
    if scheduler is not None and sched_state:
        scheduler.load_state_dict(sched_state)
    return ckpt.get("state", {}) or {}


def checkpoint_config(path: str) -> dict:
    """
    Model config of a checkpoint: stored inside the checkpoint (v0.2+), else the model_config.json next
    to it (v0.1 checkpoints).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("config"):
        return dict(ckpt["config"])
    return load_sidecar_config(path)


def load_model(path: str, device="cpu", vocab_size: int | None = None) -> CausalConvTransformerLM:
    cfg = checkpoint_config(path)
    if vocab_size is not None:
        cfg["vocab_size"] = int(vocab_size)
    model = CausalConvTransformerLM(CCTConfig.from_dict(cfg))
    load_checkpoint(path, model)
    return model.to(device).eval()


def write_sidecar_config(ckpt_dir: str, config: dict, overwrite: bool = True):
    """Write model_config.json (kept for v0.1 tools; the checkpoint itself also carries the config)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    sidecar = os.path.join(ckpt_dir, "model_config.json")
    if overwrite or not os.path.exists(sidecar):
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


def try_write_sidecar_config(ckpt_dir: str, config: dict):
    """v0.1 API: write model_config.json only if missing."""
    write_sidecar_config(ckpt_dir, config, overwrite=False)


def ckpt_dir_from_path(ckpt_path: str) -> str:
    if os.path.isdir(ckpt_path):
        return ckpt_path
    return os.path.dirname(os.path.abspath(ckpt_path))


def load_sidecar_config(ckpt_path_or_dir: str) -> dict[str, Any]:
    d = ckpt_dir_from_path(ckpt_path_or_dir)
    sidecar = os.path.join(d, "model_config.json")
    if not os.path.exists(sidecar):
        raise FileNotFoundError(f"model_config.json not found in {d}")
    with open(sidecar, encoding="utf-8") as f:
        return json.load(f)


def latest_checkpoint_in_dir(d: str) -> str | None:
    if not os.path.isdir(d):
        return None
    files = [f for f in os.listdir(d) if re.search(r"(epoch\d+|last)\.pt$", f)]
    if not files:
        return None

    def key(name):
        m = re.search(r"epoch(\d+)\.pt$", name)
        return (int(m.group(1)) if m else -1, os.path.getmtime(os.path.join(d, name)))

    files.sort(key=key, reverse=True)
    if "last.pt" in files:  # written at every save, so it is the most recent state
        return os.path.join(d, "last.pt")
    return os.path.join(d, files[0])


def resolve_resume_arg(resume: str, root: str = "checkpoints") -> str:
    if os.path.isfile(resume):
        return resume
    for cand in (resume, os.path.join(root, resume)):
        if os.path.isdir(cand):
            path = latest_checkpoint_in_dir(cand)
            if path:
                return path
            raise FileNotFoundError(f"No checkpoint files in {cand}")
    raise FileNotFoundError(f"Could not resolve resume target: {resume}")
