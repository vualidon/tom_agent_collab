# PAEC Implementation Design Spec

## Overview

**Perspective-Anchored Evidential Coordination (PAEC)** — a training-minimal framework that improves Theory of Mind reasoning in small LLMs (Qwen3-1.7B) via perspective prefix tuning + Dempster-Shafer evidential fusion.

**Core claim:** By running a small LLM twice with perspective-typed prefixes and fusing the results via subjective logic, we can close the explicit-to-applied ToM gap at 2× forward pass cost.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Base model | Qwen3-1.7B | Small, PEFT-compatible, available on HuggingFace |
| Training method | Prefix tuning (PEFT) | Minimal params, minutes to train, reusable across tasks |
| Training data | ToMi + ExploreToM (public only) | Reproducibility, no teacher model dependency |
| Primary benchmark | CoordinationQA (198 MC) | Tests coordination reasoning without game env complexity |
| Secondary benchmarks | SimpleToM + ToMi | Explicit-to-applied gap + false-belief accuracy |
| Uncertainty mapping | LogTokU-adapted logit→Dirichlet | Public code, principled evidential foundation |
| Fusion method | Dempster's combination rule | Designed for structurally distinct evidence sources |
| Answer selection | argmax(projected_prob) | projected_prob = b_fused + prior × u_fused |
| GPU target | Google Colab T4 (15GB) | Free tier, sufficient for 1.7B prefix tuning |
| Fallback if prefixes fail | LoRA adapters (rank 8-16) | More expressive, same pipeline downstream |

## Architecture

```
tom_agent_collab/
├── PAEC_revised_proposal.md
├── docs/superpowers/specs/
│   └── 2026-03-13-paec-design.md     # This file
├── src/
│   ├── data/
│   │   ├── tomi_loader.py             # ToMi dataset loading + perspective-pair formatting
│   │   ├── exploretom_loader.py       # ExploreToM dataset loading
│   │   └── prefix_training_data.py    # Combine & format training examples
│   ├── models/
│   │   ├── perspective_prefix.py      # PEFT PrefixTuningConfig for K=2 prefixes
│   │   └── model_loader.py            # Qwen3-1.7B loading with/without prefixes
│   ├── inference/
│   │   ├── perspective_runner.py      # 2-pass inference (self + partner prefix)
│   │   ├── logit_to_dirichlet.py      # LogTokU-style logit → Dirichlet mapping
│   │   └── ds_fusion.py              # Dempster-Shafer combination + vacuity/dissonance
│   ├── evaluation/
│   │   ├── coordination_qa.py         # CoordinationQA MC evaluation
│   │   ├── simpletom_eval.py          # SimpleToM evaluation
│   │   ├── tomi_eval.py               # ToMi evaluation
│   │   └── calibration.py             # ECE, Brier score
│   └── baselines/
│       ├── standard_prompting.py      # Baseline 1: vanilla prompting
│       ├── simtom_prompting.py        # Baseline 2: SimToM perspective prompts
│       └── self_consistency.py        # Baseline 3/4: SC voting (N=2, N=8)
├── notebooks/
│   ├── 01_prefix_training.ipynb       # Colab: train perspective prefixes (T4)
│   ├── 02_inference_pipeline.ipynb    # Colab: run PAEC pipeline
│   └── 03_evaluation.ipynb            # Colab: full benchmark suite
├── configs/
│   └── default.yaml                   # All hyperparameters
└── scripts/
    └── run_all_experiments.py          # Orchestrate Experiments 1-6
```

## Data Pipeline

### Training Data (Prefix Tuning)

**ToMi** (~6K examples): Programmatically generated Sally-Anne false-belief stories. Extract perspective pairs — same story reformatted for self-view and partner-view with ground-truth belief labels. 80/20 train/test split.

**ExploreToM** (~8-10K examples): 13.3K rows from HuggingFace (facebook/ExploreToM). Filter to 1st and 2nd order belief questions. Same perspective-pair extraction.

**Total:** ~15K perspective-labeled training examples.

**Data validation (before training):**

- Verify self/partner label balance (~50/50 split)
- Log example counts per source (ToMi vs ExploreToM)
- Print sequence length distribution (mean, p50, p95, max)
- Confirm ExploreToM 1st/2nd order filtering yields expected ~8-10K examples
- Spot-check 10 random examples for correct perspective labeling

**Format per example:**
```
[PREFIX_SELF or PREFIX_PARTNER]
Story: {narrative}
Question: What does {agent_name} believe about {object}?
Answer: {ground_truth_belief}
```

### Evaluation Data

- **CoordinationQA:** 198 MC questions from github.com/eric-ai-lab/llm_coordination. 3 dimensions: env comprehension, ToM reasoning, joint planning.
- **SimpleToM:** 1,147 stories / 3,441 questions from HuggingFace allenai/SimpleToM. Binary format: mental state / behavior prediction / judgment.
- **ToMi test split:** ~200 stories held out from training. QA with ground-truth belief labels.

## Core Pipeline

### Step 1 — Perspective-Anchored Generation

Two forward passes per scenario:
- Pass 1: `[PREFIX_SELF] + scenario + question → logits_self`
- Pass 2: `[PREFIX_PARTNER] + scenario + question → logits_partner`
- Temperature = 0 (greedy), collect logits over answer tokens

### Step 2 — Logit-to-Dirichlet Mapping

```python
# Center logits to prevent information loss from ReLU on negative logits
# (configurable: "center_min", "center_mean", or "softplus")
if config.logit_transform == "center_min":
    logits = logits - logits.min()
elif config.logit_transform == "center_mean":
    logits = logits - logits.mean()

if config.evidence_fn == "relu":
    evidence = relu(logits)
elif config.evidence_fn == "softplus":
    evidence = softplus(logits)  # smooth alternative, no hard zeroing

alpha = evidence + 1                          # Dirichlet concentration
S = sum(alpha)                                # Dirichlet strength
belief = (alpha - 1) / S                      # belief mass per option
uncertainty = K / S                           # vacuity
opinion = (belief, uncertainty, prior=1/K)
```

For MC: logits at answer letter position (A/B/C/D). For binary QA: logits over yes/no tokens.

**Design note:** Raw ReLU on logits can zero out all negative logits, producing pure vacuity even when the model has a clear preference. Logit centering (subtract min or mean) before ReLU preserves relative ranking. Softplus is a smooth alternative. Default: `center_min` + `relu`.

### Step 3 — Dempster-Shafer Fusion

```python
# Assumption: opinions have singleton focal elements only (guaranteed by Dirichlet mapping)
conflict = sum(b_self[i] * b_partner[j] for i != j)

# Guard against total conflict (division by zero)
CONFLICT_EPSILON = 1e-8
if conflict >= 1.0 - CONFLICT_EPSILON:
    # Fallback: average the two opinions instead of Dempster fusion
    b_fused = 0.5 * (b_self + b_partner)
    u_fused = 0.5 * (u_self + u_partner)
    normalized_conflict = 1.0  # flag as maximum conflict
else:
    K_norm = 1 - conflict
    b_fused[i] = (b_self[i]*b_partner[i] + b_self[i]*u_partner + u_self*b_partner[i]) / K_norm
    u_fused = (u_self * u_partner) / K_norm
    normalized_conflict = conflict / K_norm  # called "dissonance" in the paper narrative

projected_prob = b_fused + prior * u_fused
```

**Naming note:** `normalized_conflict` in code maps to "dissonance" in the paper. This avoids confusion with Josang's formal definition of dissonance in subjective logic.

### Step 4 — Answer Selection

```python
answer = argmax(projected_prob)
confidence = 1 - u_fused

# Degenerate opinion fallback: if fusion adds no information, use raw logits
if u_fused > 0.95:
    answer = argmax(raw_logits_self)  # fall back to single-pass answer
    confidence = 0.0  # flag as low-confidence
```

**Vacuity-guided routing:** The proposal describes a 3-way routing policy (observe / disambiguate / act) based on thresholds τ_u and τ_δ. This is deferred for the MC evaluation setting (no interactive actions available). For the paper: we demonstrate routing via qualitative case studies and accuracy-when-confident curves, not as an active policy.

## Training Configuration (Colab T4)

### Prefix Tuning

Two separate prefix models trained independently:

| Parameter | Value |
|---|---|
| num_virtual_tokens | 20 |
| prefix_projection | True |
| encoder_hidden_size | 1024 |
| Trainable params | To be verified empirically via `model.print_trainable_parameters()` |

### Optimization

| Parameter | Value |
|---|---|
| Batch size | 4 |
| Gradient accumulation | 8 (effective batch 32) |
| Learning rate | 3e-4 |
| Scheduler | Cosine, 10% warmup |
| Epochs | 3 |
| Precision | fp16 |
| Max sequence length | 512 |
| Optimizer | AdamW (8-bit via bitsandbytes) |
| Gradient checkpointing | True (required to fit in T4 VRAM) |
| Checkpoint strategy | Save after each epoch to Google Drive |

**VRAM budget:** ~10-12GB of 15GB with gradient checkpointing enabled. Without it, activations alone could exceed 15GB.

**Training time:** ~30-45 min per prefix. ~1-1.5 hours total. Both prefixes trained in a single Colab script with intermediate Drive saves to guard against session disconnects.

**Sequence length handling:** Max 512 tokens. CoordinationQA scenarios may exceed this — run token length analysis during data loading and apply right-truncation (keep question + answer options, truncate long scenario descriptions from the left). Report truncation statistics.

## Experiments

| # | Name | Benchmark | Methods | Metric |
|---|---|---|---|---|
| 1 | Main result | CoordinationQA | All 5 baselines + PAEC | Accuracy per dimension |
| 2 | Explicit-to-applied gap | SimpleToM | All 5 baselines + PAEC | Mental state / behavior / judgment accuracy |
| 3 | ToM accuracy | ToMi (test) | All 5 baselines + PAEC | False-belief accuracy |
| 4 | Calibration | SimpleToM + ToMi | PAEC vs SC-2 vs SC-8 | ECE, Brier score |
| 5 | Prefix ablation | SimpleToM | No prefix+fusion / Prompt+fusion / Full PAEC | Accuracy delta |
| 6 | Vacuity analysis | All | PAEC only | Case studies + accuracy-when-confident |

### Baselines

1. **Standard prompting:** Single pass, no prefix, argmax on raw logits
2. **SimToM prompting:** Single pass with perspective system prompt, no fusion
3. **Self-consistency N=2:** Two passes, temp=0.7, majority vote (same compute as PAEC)
4. **Self-consistency N=8:** Eight passes, temp=0.7, majority vote (4× PAEC compute)
5. **Prompt-only + DS fusion:** Two passes with perspective prompts (not prefixes), same DS fusion

### Success Criteria

**Minimum (publishable):** PAEC beats standard + SimToM on 2/3 benchmarks. Matches SC-2 on accuracy. Vacuity correlates with errors. Better ECE than SC baselines.

**Strong:** PAEC 1.7B approaches SimToM 3B+ on SimpleToM behavior prediction. Dissonance flags real perspective conflicts. Both prefixes and fusion contribute independently in ablation.

### Statistical Rigor

- Bootstrap 95% CI (1000 resamples)
- McNemar's test for pairwise comparisons
- Run 3 seeds (42, 123, 456) for prefix training; report mean +/- std on all metrics
- Set `torch.manual_seed`, `torch.cuda.manual_seed_all`, `transformers.set_seed`
- Use `torch.backends.cudnn.deterministic = True` for GPU reproducibility
- All results reproducible from seed + config YAML

## Build Order

1. **Phase 1 (Approach 2):** Prompt-only perspectives + DS fusion pipeline. Validates fusion math, evaluation harness, and baselines. No training needed.
2. **Phase 2 (Approach 1):** Add prefix tuning. Train on ToMi + ExploreToM. Compare against Phase 1 to isolate prefix contribution.
3. **Phase 3 (Fallback):** If prefixes underperform, swap in LoRA adapters and reframe as "perspective adapter tuning."

## External Dependencies

| Dependency | Source | Purpose |
|---|---|---|
| Qwen3-1.7B | HuggingFace Qwen/Qwen3-1.7B | Base model |
| PEFT | pip install peft | Prefix tuning |
| ToMi | github.com/facebookresearch/ToMi | Training data + eval |
| ExploreToM | HuggingFace facebook/ExploreToM | Training data |
| SimpleToM | HuggingFace allenai/SimpleToM | Evaluation |
| LLM-Coordination | github.com/eric-ai-lab/llm_coordination | CoordinationQA eval |
| LogTokU | github.com/MaHuanAAA/logtoku | Reference for logit→Dirichlet |
| bitsandbytes | pip install bitsandbytes | 8-bit optimizer |
| transformers | pip install transformers | Model loading |

## Config Schema (`configs/default.yaml`)

```yaml
# Model
model_name: "Qwen/Qwen3-1.7B"
max_seq_length: 512

# Prefix tuning
num_virtual_tokens: 20
prefix_projection: true
encoder_hidden_size: 1024

# Training
batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 3.0e-4
num_epochs: 3
warmup_ratio: 0.1
fp16: true
gradient_checkpointing: true
optimizer: "adamw_8bit"

# Inference
temperature: 0.0  # greedy decoding for PAEC passes
sc_temperature: 0.7  # temperature for self-consistency baselines

# Logit-to-Dirichlet
logit_transform: "center_min"  # "center_min", "center_mean", or "none"
evidence_fn: "relu"  # "relu" or "softplus"

# Fusion
conflict_epsilon: 1.0e-8  # guard against division by zero
degenerate_vacuity_threshold: 0.95  # fallback to raw logits

# Thresholds (for qualitative analysis / case studies)
tau_u: 0.7  # vacuity threshold for "high uncertainty" flag
tau_delta: 0.5  # normalized conflict threshold for "high dissonance" flag

# Evaluation
seeds: [42, 123, 456]
bootstrap_resamples: 1000

# Colab
save_to_drive: true
checkpoint_every_epoch: true
```

## Scope Clarifications

- **Cross-model scaling (Experiment 3 in proposal):** Deferred. Initial implementation targets only Qwen3-1.7B. Extension to Llama-3.2-3B and Phi-3-mini-4B is future work.
- **Vacuity-guided action routing:** Deferred for MC evaluation. Demonstrated via qualitative case studies and accuracy-when-confident curves, not as an active policy.
- **Communication quality experiment (Experiment 2 in proposal):** Deferred. Requires interactive game environments. Replaced with SimpleToM explicit-to-applied gap evaluation.
- **Prefix diversity monitoring:** Track cosine similarity between prefix_self and prefix_partner embeddings during training to detect convergence to similar representations.
