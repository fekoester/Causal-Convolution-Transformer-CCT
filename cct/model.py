"""
Causal Convolution Transformer (CCT): a GPT-style decoder whose blocks add a causal-convolution
residual branch before the (parallel) attention + MLP update:

    x <- x + s * sum_k Conv_k(x)            causal conv over the last `dyn_window + 1` tokens
    x <- x + Attn(LN1(x)) + MLP(LN2(x))     attention and MLP read the same input (GPT-J style)

Defaults reproduce v0.1 exactly (same parameters, same outputs); see CCTConfig for the options.
"""

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CCTConfig:
    vocab_size: int = 50_257
    seq_length: int = 1024  # length of the stored training windows; the model sees seq_length - 1 tokens
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dyn_window: int = 2  # each conv sees the current token and the previous `dyn_window` tokens
    dyn_layers: int = 1  # number of conv paths per block (0 = plain transformer, the ablation baseline)
    dropout: float = 0.0
    # "parallel" (v0.1): the paths all read the block input and are summed (scaled by 1/K).
    # "stacked": the paths are applied one after another (conv -> LN -> GELU -> conv ...), the reading
    # that the GELU between paths in v0.1 suggests.
    conv_mode: str = "parallel"
    # None (v0.1): fixed scale 1/K on the conv branch. A float: learnable scale initialised to this
    # value (0.0 = ReZero, the branch starts neutral).
    conv_scale_init: float | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CCTConfig":
        """Build from a (possibly older) dict, ignoring unknown keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q, self.k, self.v = (nn.Linear(n_embd, n_embd) for _ in range(3))
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # attention dropout only while training (v0.1 also applied it in eval mode)
        p = self.drop.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class TransformerBlockWithCausalConvs(nn.Module):
    """Pre-norm block: causal-conv residual branch, then parallel attention + MLP."""

    def __init__(self, n_embd, n_head, dropout, L, K=1, conv_mode="parallel", conv_scale_init=None):
        super().__init__()
        if conv_mode not in ("parallel", "stacked"):
            raise ValueError("conv_mode must be 'parallel' or 'stacked'")
        self.ln1, self.attn = nn.LayerNorm(n_embd), MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )
        self.conv_path = nn.ModuleList([nn.Conv1d(n_embd, n_embd, kernel_size=L + 1, bias=False) for _ in range(K)])
        with torch.no_grad():
            for conv in self.conv_path:
                # tap 0 is the oldest token in the window (kept from v0.1; it trains normally afterwards)
                conv.weight[:, :, 0].zero_()
        self.inter_norm, self.act, self.res_scale = nn.LayerNorm(n_embd), nn.GELU(), (1.0 / K if K > 0 else 0.0)
        self.conv_mode = conv_mode
        self.conv_gain = None if conv_scale_init is None else nn.Parameter(torch.tensor(float(conv_scale_init)))
        self._enabled = K > 0

    def _causal_conv(self, conv, H):
        pad = conv.kernel_size[0] - 1
        return conv(F.pad(H, (pad, 0)))  # left padding only: output t sees inputs t-L..t

    def conv_branch(self, x):
        H = x.transpose(1, 2)
        if self.conv_mode == "parallel":
            acc = None
            for i, conv in enumerate(self.conv_path):
                y = self.inter_norm(self._causal_conv(conv, H).transpose(1, 2))
                y = self.act(y) if i < len(self.conv_path) - 1 else y
                acc = y if acc is None else acc + y
            out = self.res_scale * acc
        else:
            for i, conv in enumerate(self.conv_path):
                y = self.inter_norm(self._causal_conv(conv, H).transpose(1, 2))
                if i < len(self.conv_path) - 1:
                    y = self.act(y)
                    H = y.transpose(1, 2)
            out = y
        return out if self.conv_gain is None else self.conv_gain * out

    def forward(self, x):
        if self._enabled:
            x = x + self.conv_branch(x)
        return x + self.attn(self.ln1(x)) + self.mlp(self.ln2(x))


class CausalConvTransformerLM(nn.Module):
    def __init__(self, config: CCTConfig):
        super().__init__()
        self.config = config
        self.seq_length = config.seq_length - 1
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, config.n_embd))
        self.blocks = nn.Sequential(
            *[
                TransformerBlockWithCausalConvs(
                    config.n_embd, config.n_head, config.dropout, config.dyn_window, config.dyn_layers,
                    config.conv_mode, config.conv_scale_init,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f, self.lm_head = nn.LayerNorm(config.n_embd), nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # GPT-2 style init: small weights help stabilize early loss
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel() + self.pos_emb.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.size()
        if T > self.seq_length:
            raise ValueError(f"sequence of length {T} is longer than the model's context ({self.seq_length})")
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None, generator=None):
        """Autoregressive sampling with top-k / nucleus filtering (context cropped to the model's length)."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.seq_length:])
            logits = logits[:, -1, :].float() / max(1e-8, temperature)
            if top_k:
                k = min(int(top_k), logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < v[:, [-1]], -math.inf)
            probs = torch.softmax(logits, dim=-1)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                mask[..., 1:] = mask[..., :-1].clone()  # keep the token that crosses top_p
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
