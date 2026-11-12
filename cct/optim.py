import math

from torch import optim


def param_groups(model, weight_decay: float):
    """
    Decay only the matrices and kernels (parameters with >= 2 dims: linear weights, conv kernels, the
    tied embedding, the position embedding); not biases, LayerNorm gains or scalar gates.
    """
    decay, no_decay, seen = [], [], set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:  # tied weights appear once
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model, lr: float, weight_decay: float = 0.1, decay_all: bool = False, fused: bool = False):
    """AdamW (betas 0.9/0.95). decay_all=True reproduces v0.1 (weight decay on every parameter)."""
    params = model.parameters() if decay_all else param_groups(model, weight_decay)
    kw = {"fused": True} if fused else {}
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), **kw)


class WarmHoldCosineLR:
    """
    Linear warmup -> hold -> cosine decay to min_lr_ratio * base_lr.

    The LR for optimizer update n (n = 0, 1, ...) is get_lr at step_num = n: the first update already
    runs at the start of the warmup, and progress is clamped at the end of the schedule (v0.1 ran the
    first update at the full LR and let the cosine rise again after the last step).
    """

    def __init__(self, optimizer, total_steps: int, warmup_frac: float, hold_frac: float, min_lr_ratio: float):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = int(self.total_steps * warmup_frac)
        self.hold_steps = int(self.total_steps * hold_frac)
        self.decay_steps = self.total_steps - self.warmup_steps - self.hold_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [g.get("initial_lr", g["lr"]) for g in optimizer.param_groups]
        self._apply()

    def _apply(self):
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = self.get_lr(base_lr)

    def step(self):
        self.step_num += 1
        self._apply()

    def get_lr(self, base_lr: float):
        if self.step_num < self.warmup_steps:
            return base_lr * (self.step_num + 1) / (self.warmup_steps + 1)
        if self.step_num < self.warmup_steps + self.hold_steps:
            return base_lr
        progress = (self.step_num - self.warmup_steps - self.hold_steps) / max(1, self.decay_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "base_lrs": self.base_lrs,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "hold_steps": self.hold_steps,
            "decay_steps": self.decay_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, state):
        self.total_steps = state.get("total_steps", self.total_steps)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.hold_steps = state.get("hold_steps", self.hold_steps)
        self.decay_steps = state.get("decay_steps", self.decay_steps)
        self.min_lr_ratio = state.get("min_lr_ratio", self.min_lr_ratio)
        self.step_num = state.get("step_num", 0)
        self.base_lrs = state.get("base_lrs", self.base_lrs)
        self._apply()
