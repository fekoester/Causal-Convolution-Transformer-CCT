import os

import numpy as np
import pytest
import torch

from cct.checkpoint import load_model
from cct.data import LMDataset
from cct.download_data import prepare_char_level
from scripts import sample, train, validate

TEXT = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n"
        "First Citizen:\nYou are all resolved rather to die than to famish?\n\n") * 120


@pytest.fixture
def text_file(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(TEXT)
    return str(p)


def _args(text_file, tmp_path, tag, *extra):
    return ["--dataset", "text", "--text-file", text_file, "--data-dir", str(tmp_path / "data"), "--tag", tag,
            "--checkpoint-root", str(tmp_path / "ck"), "--seq-length", "17", "--n-embd", "16", "--n-layer", "1",
            "--n-head", "2", "--batch-size", "4", "--num-workers", "0", "--log-every", "1", "--eval-batches", "2",
            "--use-scheduler", "--warmup-frac", "0.2", "--epochs", "1", *extra]


def test_char_prep_and_memmap_dataset(tmp_path):
    prepare_char_level(TEXT, str(tmp_path), "text")
    ds = LMDataset(str(tmp_path / "train.bin"), 17)
    raw = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    assert len(ds) == len(raw) // 17
    x, y = ds[3]
    assert x.shape == (16,) and torch.equal(x[1:], y[:-1])
    assert int(x[0]) == raw[3 * 17]


def test_mid_epoch_resume_is_exact(text_file, tmp_path):
    train.main(_args(text_file, tmp_path, "straight", "--max-steps", "6"))
    train.main(_args(text_file, tmp_path, "resumed", "--max-steps", "3"))
    train.main(_args(text_file, tmp_path, "resumed", "--max-steps", "6", "--resume", "resumed"))
    a = torch.load(tmp_path / "ck" / "straight" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "ck" / "resumed" / "last.pt", weights_only=False)
    assert a["state"]["step"] == b["state"]["step"] == 6
    for k, v in a["model"].items():
        torch.testing.assert_close(v, b["model"][k], msg=k)
    assert a["scheduler"]["step_num"] == b["scheduler"]["step_num"] == 6


def test_epoch_end_eval_best_logs_validate_and_sample(text_file, tmp_path):
    out = train.main(_args(text_file, tmp_path, "run"))
    run = tmp_path / "ck" / "run"
    assert np.isfinite(out["best_val"])
    for f in ("best.pt", "epoch1.pt", "last.pt", "train_log.csv", "val_log.csv", "model_config.json"):
        assert (run / f).exists(), f
    loss = validate.main(["--ckpt", str(run / "best.pt"), "--data-dir", str(tmp_path / "data"), "--max-batches", "3"])
    assert loss > 0
    (text,) = sample.main(["--ckpt", str(run / "best.pt"), "--data-dir", str(tmp_path / "data"), "--prompt",
                           "First", "--max-new-tokens", "10", "--seed", "0", "--top-k", "500"])
    assert text.startswith("First") and len(text) == 15


def test_v01_checkpoint_without_config_loads_via_sidecar(text_file, tmp_path):
    train.main(_args(text_file, tmp_path, "old", "--max-steps", "1"))
    run = tmp_path / "ck" / "old"
    ck = torch.load(run / "last.pt", weights_only=False)
    ck.pop("config")
    torch.save(ck, run / "epoch1.pt")
    m = load_model(str(run / "epoch1.pt"))
    assert m.config.n_embd == 16 and os.path.exists(run / "model_config.json")
