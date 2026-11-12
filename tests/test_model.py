import importlib.util
import pathlib
import subprocess
import sys

import pytest
import torch

from cct.model import CausalConvTransformerLM, CCTConfig
from cct.optim import WarmHoldCosineLR, build_optimizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMALL = dict(vocab_size=50, seq_length=33, n_embd=32, n_layer=2, n_head=4)


def _original_model():
    try:
        first = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.split()[0]
        code = subprocess.run(["git", "show", f"{first}:cct/model.py"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None
    path = pathlib.Path(__file__).parent / "_orig_model_tmp.py"
    path.write_text(code)
    try:
        spec = importlib.util.spec_from_file_location("_orig_model_tmp", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        path.unlink()
    return mod


@pytest.mark.parametrize("K", [0, 1, 2])
def test_defaults_match_v01(K):
    orig = _original_model()
    if orig is None:
        pytest.skip("needs git history")
    torch.manual_seed(0)
    a = orig.CausalConvTransformerLM(orig.CCTConfig(**SMALL, dyn_layers=K)).eval()
    torch.manual_seed(0)
    b = CausalConvTransformerLM(CCTConfig(**SMALL, dyn_layers=K)).eval()
    assert list(a.state_dict()) == list(b.state_dict())
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    x = torch.randint(0, 50, (2, 32))
    torch.testing.assert_close(a(x, x)[0], b(x, x)[0])


@pytest.mark.parametrize("kw", [dict(), dict(dyn_layers=2, conv_mode="stacked"), dict(dyn_window=6),
                                dict(conv_scale_init=0.5)])
def test_causality(kw):
    """Changing token t must not change any logits before t (conv branch and attention)."""
    torch.manual_seed(0)
    m = CausalConvTransformerLM(CCTConfig(**SMALL, **kw)).eval()
    with torch.no_grad():  # make the conv branch matter
        for blk in m.blocks:
            for conv in blk.conv_path:
                conv.weight.normal_(std=0.1)
    x = torch.randint(0, 50, (1, 32))
    x2 = x.clone()
    x2[0, 20] = (x2[0, 20] + 1) % 50
    a, b = m(x)[0], m(x2)[0]
    torch.testing.assert_close(a[:, :20], b[:, :20])
    assert (a[:, 20:] - b[:, 20:]).abs().max() > 1e-6


def test_rezero_branch_starts_neutral():
    torch.manual_seed(0)
    plain = CausalConvTransformerLM(CCTConfig(**SMALL, dyn_layers=0)).eval()
    torch.manual_seed(0)
    gated = CausalConvTransformerLM(CCTConfig(**SMALL, conv_scale_init=0.0)).eval()
    gated.load_state_dict(plain.state_dict(), strict=False)  # same attention/MLP weights
    x = torch.randint(0, 50, (2, 32))
    torch.testing.assert_close(gated(x)[0], plain(x)[0])


def test_attention_dropout_off_in_eval():
    torch.manual_seed(0)
    m = CausalConvTransformerLM(CCTConfig(**SMALL, dropout=0.5)).eval()
    x = torch.randint(0, 50, (2, 32))
    torch.testing.assert_close(m(x)[0], m(x)[0])


def test_weight_decay_groups():
    m = CausalConvTransformerLM(CCTConfig(**SMALL, conv_scale_init=0.0))
    opt = build_optimizer(m, 1e-3, 0.1)
    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    n_unique = len({id(p) for p in m.parameters()})
    assert len(decay["params"]) + len(no_decay["params"]) == n_unique  # tied weight counted once
    legacy = build_optimizer(m, 1e-3, 0.1, decay_all=True)
    assert len(legacy.param_groups) == 1 and legacy.param_groups[0]["weight_decay"] == 0.1


def test_scheduler_warmup_starts_low_and_never_rises_at_the_end():
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    s = WarmHoldCosineLR(opt, total_steps=100, warmup_frac=0.1, hold_frac=0.0, min_lr_ratio=0.1)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(120):
        s.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < 0.2  # the first update is inside the warmup
    assert max(lrs) == pytest.approx(1.0)
    assert all(b <= a + 1e-12 for a, b in zip(lrs[10:], lrs[11:]))  # monotone after warmup
    assert lrs[-1] == pytest.approx(0.1)


def test_generate_and_long_input():
    torch.manual_seed(0)
    m = CausalConvTransformerLM(CCTConfig(**SMALL)).eval()
    out = m.generate(torch.zeros(1, 40, dtype=torch.long), max_new_tokens=5, top_k=1000, top_p=0.9)
    assert out.shape == (1, 45)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 40, dtype=torch.long))
