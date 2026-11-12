# Causal Convolution Transformer (CCT)

CCT is a compact GPT-style language model that adds a **causal convolution residual path** to each transformer block. The convolution gives every block a cheap, explicit view of the last few tokens. Attention can then spend its capacity on longer-range structure.

This repo contains a minimal training pipeline:

- CPU/GPU auto-detection, with bf16 autocast on CUDA (or `--amp-dtype fp16` with loss scaling) and optional `torch.compile`
- three datasets out of the box:
  - **`tiny`**: tiny Shakespeare, character level
  - **`text`**: any local text file, character level
  - **`openwebtext`**: GPT-2 BPE, streamed and written to disk in chunks
- one command each to **train**, **resume** (exactly, even mid-epoch), **validate**, **sample** and **plot**
- validation during training, `best.pt`, and CSV logs
- an ablation script: CCT against the same model without the convolution
- no package install required; run with `python -m scripts.*`

## Architecture

Each block updates the residual stream $x \in \mathbb{R}^{T\times d}$ in two steps:

$$
\begin{aligned}
x &\leftarrow x + s\sum_{k=1}^{K}\phi_k\Big(\mathrm{LN}\big(\mathrm{Conv}^{\text{causal}}_k(x)\big)\Big) && \text{conv branch over the last } L{+}1 \text{ tokens} \\
x &\leftarrow x + \mathrm{Attn}\big(\mathrm{LN}_1(x)\big) + \mathrm{MLP}\big(\mathrm{LN}_2(x)\big) && \text{attention and MLP in parallel (GPT-J style)}
\end{aligned}
$$

- $\mathrm{Conv}^{\text{causal}}_k$ is a `Conv1d` with kernel size $L+1$ (`--dyn-window L`). Its input is padded on the left only, so the output at position $t$ depends only on $x_{t-L}, \dots, x_t$; a test checks this.
- $\phi_k$ is a GELU for all but the last path.
- $s = 1/K$ by default (`--dyn-layers K`).

Options, whose defaults reproduce v0.1 exactly:

| flag | effect |
|---|---|
| `--dyn-layers 0` | no conv branch: the plain-transformer baseline for ablations |
| `--conv-mode stacked` | the K convolutions are applied one after another instead of summed in parallel |
| `--conv-scale-init 0` | the conv branch gets a learnable scale starting at 0 (ReZero), so the block starts as a plain transformer. By default the LayerNormed branch adds a unit-variance vector per layer from the first step. |

Weights are tied between the embedding and the output layer, with GPT-2 initialisation (std 0.02) and learned absolute positions. The context is `seq_length − 1` tokens.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# or CUDA 12.4:  pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Quickstart

### Tiny Shakespeare

```bash
# train (validation every 500 updates, best.pt kept)
python -m scripts.train --dataset tiny --data-dir data/tiny --tag tiny_demo \
    --epochs 2 --seq-length 256 --n-embd 128 --n-layer 4 --n-head 4 \
    --batch-size 16 --lr 3e-4 --use-scheduler --grad-clip 1.0 --eval-every 500

# continue the same run (restores model, optimizer, LR schedule, RNG and position)
python -m scripts.train --dataset tiny --data-dir data/tiny --tag tiny_demo --epochs 2 \
    --use-scheduler --resume tiny_demo

# or train 2 more epochs with a fresh LR cycle over them
python -m scripts.train --dataset tiny --data-dir data/tiny --tag tiny_demo \
    --use-scheduler --resume tiny_demo --epochs-add 2

python -m scripts.validate --ckpt checkpoints/tiny_demo/best.pt --data-dir data/tiny
python -m scripts.sample --ckpt checkpoints/tiny_demo/best.pt --data-dir data/tiny \
    --prompt "To be, or not to be," --temperature 0.8 --top-k 50 --top-p 0.9 --max-new-tokens 200
python -m scripts.plot_loss checkpoints/tiny_demo --out loss.png
```

The model config travels inside every checkpoint, so `validate`, `sample` and `--resume` always rebuild exactly the trained model. `model_config.json` is still written next to the checkpoints, and v0.1 checkpoints load through it.

### Your own text

```bash
python -m scripts.train --dataset text --text-file my_corpus.txt --data-dir data/mine --tag mine ...
```

### OpenWebText

```bash
# 1% subset smoke test
python -m scripts.train --dataset openwebtext --subset 'train[:1%]' --data-dir data/owt_1pct --tag owt_1pct \
    --epochs 1 --seq-length 256 --n-embd 256 --n-layer 6 --n-head 8 \
    --batch-size 8 --lr 3e-4 --use-scheduler --grad-clip 1.0 --eval-every 1000 --save-every 1000

# the full corpus: omit --subset (or limit with --max-docs N). It is streamed and tokenized
# once into data/<dir>/train.bin (~18 GB for ~9B tokens); later runs reuse the .bin files.
```

Documents are separated by GPT-2's end-of-text token. `--save-every N` writes `last.pt` every N updates, so a long run can be stopped and resumed at any point.

## Ablation: does the convolution help?

```bash
python -m scripts.benchmark --max-steps 3000                 # one seed
python -m scripts.benchmark --max-steps 5000 --seeds 0 1 2   # with error bars
```

This trains the same 6-layer, 192-wide model on tiny Shakespeare in five variants:
- a plain transformer
- CCT as in v0.1
- CCT with a ReZero-gated conv branch
- CCT with two stacked convolutions
- CCT with a window of 8

All variants get the same budget, and the script reports the best validation bits per character as `results/ablation/results.csv` and a Markdown table. It is resumable.

## Results

An earlier run of a GPT-2-scale CCT on OpenWebText reached a validation loss of about 2.9 after 5 epochs (curve below). Its exact configuration was not recorded, so it isn't reproducible from this README. For a like-for-like answer to "does the convolution help?", use the ablation above.

![Training Loss Curve](assets/loss.png)

## Tests

```bash
pip install pytest && pytest      # tiny models only, a few seconds
```

The tests check:
- the default model is bit-identical to v0.1
- the conv branch and attention are causal
- the ReZero branch starts neutral
- dropout is off in evaluation
- the weight-decay groups
- the LR schedule
- the memory-mapped dataset
- resuming mid-epoch gives exactly the same weights as training straight through
- validate and sample work, and v0.1 checkpoints still load

## Changes in 0.2

**Fixes:**
- **OpenWebText preparation:**
  - tokens are streamed to disk instead of kept in one Python list, which could not hold the full corpus
  - the data comes from the Parquet copy on the Hub, since the script-based `openwebtext` loader no longer works with recent `datasets`
  - documents are separated by end-of-text tokens
- **Resume:** it restarted the LR schedule. It now restores the scheduler, RNG and the position inside the epoch.
- **Model config:** stored inside each checkpoint. The sidecar file is no longer left stale when a tag is reused.
- **`--seq-length` overrides removed from validate and sample.** They could not load, because the position embedding has a fixed size.
- **Attention dropout** is now off in eval mode.
- **Weight decay** no longer applies to biases, LayerNorm gains or scalar gates; `--decay-all` restores the old behaviour.
- **LR schedule:** the first update now starts inside the warmup, and the cosine no longer rises again after the last step.
- **Logging:** the printed loss no longer depends on device or `--accum-steps`.
- **`plot_loss.py`** read CSV files that were never written. Training now writes `train_log.csv` and `val_log.csv`.
- **Datasets** are memory-mapped instead of loaded into RAM.

**New:**
- validation during training, `best.pt`, `--save-every`, `--max-steps`
- `--conv-mode`, `--conv-scale-init`
- `--dataset text`, `--amp-dtype`, `--compile`
- `model.generate()` with a top-k guard
- `scripts/benchmark.py`
- tests, CI, pyproject, trimmed requirements, tiktoken for GPT-2 BPE
