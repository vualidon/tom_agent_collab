# Local Inference on Mac (MPS)

Run PAEC inference locally on macOS with Apple Silicon GPU (`mps`).

## 1) Install minimal inference dependencies

```bash
pip install "torch>=2.1.0" "transformers>=4.40.0" "peft>=0.10.0" "datasets>=2.19.0" \
  "pyyaml>=6.0" "numpy>=1.24.0" "scipy>=1.10.0" "scikit-learn>=1.3.0" \
  "tqdm>=4.65.0" "accelerate>=0.28.0" "huggingface_hub>=0.23.0"
```

## 2) Run inference using your uploaded adapters

```bash
python scripts/run_inference.py \
  --config configs/modal_budget.yaml \
  --device mps \
  --hf_repo_id thangvip/paec-qwen3-1.7b-prefixes
```

This runs:
- Phase 1: prompt-only baselines + PAEC(prompt)
- Phase 2: PAEC(prefix) using `prefix_self` and `prefix_partner` downloaded from the HF repo

## 3) Optional: use local adapter folders instead of HF

```bash
python scripts/run_inference.py \
  --config configs/modal_budget.yaml \
  --device mps \
  --prefix_self /path/to/prefix_self \
  --prefix_partner /path/to/prefix_partner
```
