# Modal Training Runbook (PAEC, 1 seed)

This runbook uses `scripts/modal_train_and_push.py`.

## 1) One-time setup

```bash
pip install -U modal
modal setup
```

Create Hugging Face secret in Modal (required by training function):

```bash
modal secret create huggingface HF_TOKEN=hf_xxx_your_write_token
```

## 2) Preflight benchmark (required first)

Runs short GPU trials on candidate `(batch_size:grad_accum)` pairs and chooses the fastest non-OOM config under a VRAM safety threshold.

```bash
modal run scripts/modal_train_and_push.py \
  --mode preflight \
  --gpu A10 \
  --config-path configs/modal_budget.yaml \
  --seed 42 \
  --preflight-steps 40 \
  --preflight-examples 512 \
  --candidates "4:8,8:4,16:2,32:1"
```

Notes:
- `--gpu` supports `T4`, `L4`, `A10`.
- Output includes `best` candidate + estimated hourly cost.

## 3) Full training (1 seed) + push to HF

This command automatically reruns preflight, uses the selected config, trains both `prefix_self` and `prefix_partner`, then pushes to Hugging Face.

```bash
modal run scripts/modal_train_and_push.py \
  --mode train \
  --gpu A10 \
  --config-path configs/modal_budget.yaml \
  --seed 42 \
  --hf-repo-id YOUR_HF_USERNAME/paec-qwen3-1.7b-prefixes \
  --preflight-steps 40 \
  --preflight-examples 512 \
  --candidates "4:8,8:4,16:2,32:1"
```

## 4) Artifacts

Saved in Modal Volume `paec-artifacts`:
- `/vol/artifacts/runs/<run_name>/prefix_self`
- `/vol/artifacts/runs/<run_name>/prefix_partner`
- `/vol/artifacts/runs/<run_name>/run_summary.json`

HF upload includes:
- `prefix_self/*` (excluding `epoch_*`)
- `prefix_partner/*` (excluding `epoch_*`)
- `runs/run_summary.json`

## 5) Budget controls

- Keep `seed=42` only for now (single-seed run).
- Start with `A10` for strong throughput/cost balance.
- Keep preflight enabled before each full run.
- If costs are tight, lower candidates (e.g., remove `32:1`) or switch to `L4`.
