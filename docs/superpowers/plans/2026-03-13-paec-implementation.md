# PAEC Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PAEC — a perspective prefix tuning + Dempster-Shafer fusion framework that improves Theory of Mind in Qwen3-1.7B, evaluated on CoordinationQA, SimpleToM, and ToMi benchmarks.

**Architecture:** Phase 1 builds the prompt-only DS fusion pipeline (no training). Phase 2 adds prefix tuning. Phase 3 runs all experiments. Each phase produces working, testable software independently.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers/PEFT/Datasets, bitsandbytes, Google Colab (T4 GPU), YAML configs.

**Spec:** `docs/superpowers/specs/2026-03-13-paec-design.md`

---

## File Structure

```
tom_agent_collab/
├── configs/
│   └── default.yaml                    # All hyperparameters
├── src/
│   ├── __init__.py
│   ├── config.py                       # YAML config loader (dataclass)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tomi_loader.py              # ToMi dataset: generate, parse, perspective-pair
│   │   ├── exploretom_loader.py        # ExploreToM: load from HF, filter, perspective-pair
│   │   └── prefix_training_data.py     # Combine sources, validate, create DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py             # Load Qwen3-1.7B (fp16, optional prefix/LoRA)
│   │   └── perspective_prefix.py       # PrefixTuningConfig builder, train loop, save/load
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── logit_to_dirichlet.py       # Logit centering + evidence fn → opinion
│   │   ├── ds_fusion.py                # Dempster combination + vacuity/dissonance
│   │   └── perspective_runner.py       # Orchestrate 2-pass inference → fused answer
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── coordination_qa.py          # CoordinationQA loader + MC evaluator
│   │   ├── simpletom_eval.py           # SimpleToM loader + 3-question evaluator
│   │   ├── tomi_eval.py                # ToMi test-split evaluator
│   │   └── calibration.py              # ECE, Brier score, accuracy-when-confident
│   └── baselines/
│       ├── __init__.py
│       ├── standard_prompting.py       # Baseline 1: single pass, argmax
│       ├── simtom_prompting.py         # Baseline 2: SimToM perspective prompt
│       └── self_consistency.py         # Baseline 3/4: SC voting (N=2, N=8)
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_logit_to_dirichlet.py
│   ├── test_ds_fusion.py
│   ├── test_perspective_runner.py
│   ├── test_calibration.py
│   ├── test_tomi_loader.py
│   ├── test_exploretom_loader.py
│   └── test_baselines.py
├── notebooks/
│   ├── 01_prefix_training.ipynb        # Colab: train both prefixes on T4
│   ├── 02_inference_pipeline.ipynb     # Colab: run PAEC + baselines
│   └── 03_evaluation.ipynb            # Colab: full experiment suite + plots
├── scripts/
│   └── run_all_experiments.py          # CLI orchestrator for all experiments
├── requirements.txt
└── setup.py
```

---

## Chunk 1: Foundation — Config, Core Math, and Tests

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/inference/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `src/baselines/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```txt
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
datasets>=2.19.0
bitsandbytes>=0.43.0
pyyaml>=6.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
tqdm>=4.65.0
accelerate>=0.28.0
```

- [ ] **Step 2: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="paec",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
```

- [ ] **Step 3: Create all `__init__.py` files**

All empty files. Create them for: `src/`, `src/data/`, `src/models/`, `src/inference/`, `src/evaluation/`, `src/baselines/`, `tests/`.

- [ ] **Step 4: Commit**

```bash
git init
git add requirements.txt setup.py src/ tests/
git commit -m "chore: scaffold project structure"
```

---

### Task 2: Config loader

**Files:**
- Create: `configs/default.yaml`
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the config YAML**

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
temperature: 0.0
sc_temperature: 0.7

# Logit-to-Dirichlet
logit_transform: "center_min"
evidence_fn: "relu"

# Fusion
conflict_epsilon: 1.0e-8
degenerate_vacuity_threshold: 0.95

# Thresholds
tau_u: 0.7
tau_delta: 0.5

# Evaluation
seeds: [42, 123, 456]
bootstrap_resamples: 1000

# Colab
save_to_drive: true
checkpoint_every_epoch: true
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config.py
import pytest
from src.config import PAECConfig

def test_load_default_config():
    cfg = PAECConfig.from_yaml("configs/default.yaml")
    assert cfg.model_name == "Qwen/Qwen3-1.7B"
    assert cfg.max_seq_length == 512
    assert cfg.num_virtual_tokens == 20
    assert cfg.conflict_epsilon == 1e-8
    assert cfg.seeds == [42, 123, 456]

def test_config_override():
    cfg = PAECConfig.from_yaml("configs/default.yaml", overrides={"batch_size": 8})
    assert cfg.batch_size == 8

def test_config_invalid_evidence_fn():
    with pytest.raises(ValueError):
        PAECConfig.from_yaml("configs/default.yaml", overrides={"evidence_fn": "invalid"})
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with "cannot import name 'PAECConfig'"

- [ ] **Step 4: Write implementation**

```python
# src/config.py
from dataclasses import dataclass, field, fields
from typing import List
import yaml

VALID_LOGIT_TRANSFORMS = {"center_min", "center_mean", "none"}
VALID_EVIDENCE_FNS = {"relu", "softplus"}

@dataclass
class PAECConfig:
    # Model
    model_name: str = "Qwen/Qwen3-1.7B"
    max_seq_length: int = 512

    # Prefix tuning
    num_virtual_tokens: int = 20
    prefix_projection: bool = True
    encoder_hidden_size: int = 1024

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    fp16: bool = True
    gradient_checkpointing: bool = True
    optimizer: str = "adamw_8bit"

    # Inference
    temperature: float = 0.0
    sc_temperature: float = 0.7

    # Logit-to-Dirichlet
    logit_transform: str = "center_min"
    evidence_fn: str = "relu"

    # Fusion
    conflict_epsilon: float = 1e-8
    degenerate_vacuity_threshold: float = 0.95

    # Thresholds
    tau_u: float = 0.7
    tau_delta: float = 0.5

    # Evaluation
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    bootstrap_resamples: int = 1000

    # Colab
    save_to_drive: bool = True
    checkpoint_every_epoch: bool = True

    def __post_init__(self):
        if self.logit_transform not in VALID_LOGIT_TRANSFORMS:
            raise ValueError(f"logit_transform must be one of {VALID_LOGIT_TRANSFORMS}")
        if self.evidence_fn not in VALID_EVIDENCE_FNS:
            raise ValueError(f"evidence_fn must be one of {VALID_EVIDENCE_FNS}")

    @classmethod
    def from_yaml(cls, path: str, overrides: dict = None) -> "PAECConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        if overrides:
            data.update(overrides)
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add configs/default.yaml src/config.py tests/test_config.py
git commit -m "feat: add config loader with YAML support and validation"
```

---

### Task 3: Logit-to-Dirichlet mapping

**Files:**
- Create: `src/inference/logit_to_dirichlet.py`
- Create: `tests/test_logit_to_dirichlet.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_logit_to_dirichlet.py
import pytest
import torch
from src.inference.logit_to_dirichlet import logits_to_opinion

def test_uniform_logits_high_vacuity():
    """Equal logits → equal belief, moderate vacuity."""
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    opinion = logits_to_opinion(logits, logit_transform="none", evidence_fn="relu")
    assert opinion.belief.shape == (4,)
    assert abs(opinion.belief.sum().item() + opinion.vacuity - 1.0) < 1e-5
    # Uniform logits: all beliefs should be equal
    assert torch.allclose(opinion.belief, opinion.belief[0].expand(4), atol=1e-5)

def test_confident_logits_low_vacuity():
    """One dominant logit → high belief on that class, low vacuity."""
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    opinion = logits_to_opinion(logits, logit_transform="none", evidence_fn="relu")
    assert opinion.belief[0] > 0.5
    assert opinion.vacuity < 0.5

def test_all_negative_logits_center_min():
    """All negative logits with center_min → still produces meaningful evidence."""
    logits = torch.tensor([-5.0, -3.0, -4.0, -6.0])
    opinion = logits_to_opinion(logits, logit_transform="center_min", evidence_fn="relu")
    # After centering: [1, 3, 2, 0] → highest belief at index 1
    assert opinion.belief[1] > opinion.belief[0]
    assert opinion.vacuity < 1.0  # Not pure vacuity

def test_all_negative_logits_no_centering_pure_vacuity():
    """All negative logits without centering → ReLU zeros all → pure vacuity."""
    logits = torch.tensor([-5.0, -3.0, -4.0, -6.0])
    opinion = logits_to_opinion(logits, logit_transform="none", evidence_fn="relu")
    # ReLU zeros everything → alpha = [1,1,1,1] → vacuity = K/S = 4/4 = 1.0
    assert opinion.vacuity > 0.99

def test_softplus_no_hard_zeros():
    """Softplus never produces zero evidence."""
    logits = torch.tensor([-5.0, -3.0, -4.0, -6.0])
    opinion = logits_to_opinion(logits, logit_transform="none", evidence_fn="softplus")
    assert opinion.vacuity < 1.0  # Softplus gives small positive evidence

def test_opinion_constraint():
    """belief.sum() + vacuity == 1.0 (subjective logic constraint)."""
    for _ in range(10):
        logits = torch.randn(4)
        opinion = logits_to_opinion(logits, logit_transform="center_min", evidence_fn="relu")
        assert abs(opinion.belief.sum().item() + opinion.vacuity - 1.0) < 1e-5

def test_binary_opinion():
    """Works for K=2 (binary yes/no)."""
    logits = torch.tensor([3.0, -1.0])
    opinion = logits_to_opinion(logits, logit_transform="center_min", evidence_fn="relu")
    assert opinion.belief.shape == (2,)
    assert opinion.belief[0] > opinion.belief[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_logit_to_dirichlet.py -v`
Expected: FAIL with "cannot import name 'logits_to_opinion'"

- [ ] **Step 3: Write implementation**

```python
# src/inference/logit_to_dirichlet.py
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class Opinion:
    belief: torch.Tensor     # shape (K,) — belief mass per class
    vacuity: float           # scalar — epistemic uncertainty
    alpha: torch.Tensor      # shape (K,) — Dirichlet concentration params
    prior: float             # 1/K — base rate

def logits_to_opinion(
    logits: torch.Tensor,
    logit_transform: str = "center_min",
    evidence_fn: str = "relu",
) -> Opinion:
    """Convert raw logits to a subjective logic opinion via Dirichlet mapping.

    Args:
        logits: shape (K,) — raw logits over K answer options
        logit_transform: "center_min", "center_mean", or "none"
        evidence_fn: "relu" or "softplus"
    """
    K = logits.shape[0]

    # Step 1: Center logits
    if logit_transform == "center_min":
        logits = logits - logits.min()
    elif logit_transform == "center_mean":
        logits = logits - logits.mean()

    # Step 2: Convert to non-negative evidence
    if evidence_fn == "relu":
        evidence = F.relu(logits)
    elif evidence_fn == "softplus":
        evidence = F.softplus(logits)
    else:
        raise ValueError(f"Unknown evidence_fn: {evidence_fn}")

    # Step 3: Dirichlet parameters
    alpha = evidence + 1.0
    S = alpha.sum()

    # Step 4: Subjective logic opinion
    belief = (alpha - 1.0) / S
    vacuity = float(K / S)
    prior = 1.0 / K

    return Opinion(belief=belief, vacuity=vacuity, alpha=alpha, prior=prior)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_logit_to_dirichlet.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/inference/logit_to_dirichlet.py tests/test_logit_to_dirichlet.py
git commit -m "feat: logit-to-Dirichlet mapping with centering and softplus options"
```

---

### Task 4: Dempster-Shafer fusion

**Files:**
- Create: `src/inference/ds_fusion.py`
- Create: `tests/test_ds_fusion.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ds_fusion.py
import pytest
import torch
from src.inference.logit_to_dirichlet import Opinion
from src.inference.ds_fusion import dempster_combine, FusedResult

def _make_opinion(belief: list, vacuity: float) -> Opinion:
    b = torch.tensor(belief)
    K = len(belief)
    alpha = b * 10 + 1  # dummy alpha
    return Opinion(belief=b, vacuity=vacuity, alpha=alpha, prior=1.0 / K)

def test_agreeing_opinions():
    """Two opinions that agree → high fused belief, low vacuity."""
    o1 = _make_opinion([0.7, 0.1], vacuity=0.2)
    o2 = _make_opinion([0.6, 0.1], vacuity=0.3)
    result = dempster_combine(o1, o2)
    assert result.projected_prob[0] > result.projected_prob[1]
    assert result.vacuity < min(o1.vacuity, o2.vacuity)

def test_conflicting_opinions():
    """Two opinions that disagree → high normalized_conflict."""
    o1 = _make_opinion([0.8, 0.0], vacuity=0.2)
    o2 = _make_opinion([0.0, 0.8], vacuity=0.2)
    result = dempster_combine(o1, o2)
    assert result.normalized_conflict > 0.3

def test_total_conflict_fallback():
    """Near-total conflict triggers averaging fallback."""
    o1 = _make_opinion([0.99, 0.0], vacuity=0.01)
    o2 = _make_opinion([0.0, 0.99], vacuity=0.01)
    result = dempster_combine(o1, o2, conflict_epsilon=1e-8)
    # Should not crash (no division by zero)
    assert result.belief.shape == (2,)
    assert result.normalized_conflict >= 0.99

def test_one_vacuous_opinion():
    """If one opinion is vacuous, fused result ≈ the other opinion."""
    o1 = _make_opinion([0.8, 0.0], vacuity=0.2)
    o2 = _make_opinion([0.0, 0.0], vacuity=1.0)
    result = dempster_combine(o1, o2)
    # Vacuous opinion contributes no evidence — result dominated by o1
    assert result.projected_prob[0] > result.projected_prob[1]

def test_fused_constraint():
    """belief.sum() + vacuity ≈ 1.0."""
    o1 = _make_opinion([0.5, 0.2], vacuity=0.3)
    o2 = _make_opinion([0.3, 0.4], vacuity=0.3)
    result = dempster_combine(o1, o2)
    assert abs(result.belief.sum().item() + result.vacuity - 1.0) < 1e-4

def test_four_class_fusion():
    """Works for K=4 (MC with 4 options)."""
    o1 = _make_opinion([0.5, 0.1, 0.1, 0.0], vacuity=0.3)
    o2 = _make_opinion([0.4, 0.2, 0.0, 0.1], vacuity=0.3)
    result = dempster_combine(o1, o2)
    assert result.belief.shape == (4,)
    assert result.projected_prob[0] > result.projected_prob[3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ds_fusion.py -v`
Expected: FAIL with "cannot import name 'dempster_combine'"

- [ ] **Step 3: Write implementation**

```python
# src/inference/ds_fusion.py
from dataclasses import dataclass
import torch
from src.inference.logit_to_dirichlet import Opinion

@dataclass
class FusedResult:
    belief: torch.Tensor          # shape (K,) — fused belief mass
    vacuity: float                # fused vacuity
    normalized_conflict: float    # "dissonance" in paper
    projected_prob: torch.Tensor  # shape (K,) — final answer distribution
    confidence: float             # 1 - vacuity

def dempster_combine(
    o1: Opinion,
    o2: Opinion,
    conflict_epsilon: float = 1e-8,
) -> FusedResult:
    """Combine two subjective logic opinions via Dempster's rule.

    Assumes singleton focal elements only (guaranteed by Dirichlet mapping).
    Falls back to averaging when conflict ≈ 1.0.
    """
    K = o1.belief.shape[0]
    b1, u1 = o1.belief, o1.vacuity
    b2, u2 = o2.belief, o2.vacuity
    prior = 1.0 / K

    # Compute conflict: sum of b1[i] * b2[j] for i != j
    conflict = 0.0
    for i in range(K):
        for j in range(K):
            if i != j:
                conflict += b1[i].item() * b2[j].item()

    # Guard against total conflict
    if conflict >= 1.0 - conflict_epsilon:
        b_fused = 0.5 * (b1 + b2)
        u_fused = 0.5 * (u1 + u2)
        norm_conflict = 1.0
    else:
        K_norm = 1.0 - conflict
        b_fused = torch.zeros(K)
        for i in range(K):
            b_fused[i] = (b1[i] * b2[i] + b1[i] * u2 + u1 * b2[i]) / K_norm
        u_fused = (u1 * u2) / K_norm
        norm_conflict = conflict / K_norm

    projected_prob = b_fused + prior * u_fused
    confidence = 1.0 - u_fused

    return FusedResult(
        belief=b_fused,
        vacuity=u_fused,
        normalized_conflict=norm_conflict,
        projected_prob=projected_prob,
        confidence=confidence,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ds_fusion.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/inference/ds_fusion.py tests/test_ds_fusion.py
git commit -m "feat: Dempster-Shafer fusion with conflict guard and averaging fallback"
```

---

### Task 5: Calibration metrics

**Files:**
- Create: `src/evaluation/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibration.py
import pytest
import numpy as np
from src.evaluation.calibration import expected_calibration_error, brier_score

def test_ece_perfect_calibration():
    """Perfectly calibrated predictions → ECE ≈ 0."""
    probs = np.array([0.9, 0.9, 0.1, 0.1])
    correct = np.array([1, 1, 0, 0])
    ece = expected_calibration_error(probs, correct, n_bins=2)
    assert ece < 0.1

def test_ece_worst_calibration():
    """Maximally miscalibrated → high ECE."""
    probs = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([0, 0, 0, 0])
    ece = expected_calibration_error(probs, correct, n_bins=2)
    assert ece > 0.5

def test_brier_perfect():
    """Perfect predictions → Brier = 0."""
    probs = np.array([1.0, 1.0, 0.0])
    correct = np.array([1, 1, 0])
    assert brier_score(probs, correct) < 1e-5

def test_brier_worst():
    """Worst predictions → Brier = 1."""
    probs = np.array([0.0, 0.0, 1.0])
    correct = np.array([1, 1, 0])
    assert brier_score(probs, correct) > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_calibration.py -v`
Expected: FAIL with "cannot import name 'expected_calibration_error'"

- [ ] **Step 3: Write implementation**

```python
# src/evaluation/calibration.py
import numpy as np
from typing import Tuple

def expected_calibration_error(
    probs: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        probs: predicted probabilities for the chosen answer, shape (N,)
        correct: binary correctness labels, shape (N,)
        n_bins: number of calibration bins
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = probs[mask].mean()
        bin_weight = mask.sum() / total
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)

def brier_score(probs: np.ndarray, correct: np.ndarray) -> float:
    """Compute Brier score (mean squared error between probs and correctness)."""
    return float(np.mean((probs - correct) ** 2))

def bootstrap_ci(
    metric_fn,
    probs: np.ndarray,
    correct: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.

    Returns: (mean, lower, upper)
    """
    rng = np.random.default_rng(42)
    n = len(probs)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(probs[idx], correct[idx]))

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return float(scores.mean()), float(np.percentile(scores, alpha * 100)), float(np.percentile(scores, (1 - alpha) * 100))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_calibration.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/calibration.py tests/test_calibration.py
git commit -m "feat: calibration metrics (ECE, Brier, bootstrap CI)"
```

---

## Chunk 2: Data Loaders and Benchmark Evaluators

### Task 6: ToMi data loader

**Files:**
- Create: `src/data/tomi_loader.py`
- Create: `tests/test_tomi_loader.py`

- [ ] **Step 1: Clone ToMi repository**

```bash
cd /Users/quyetthang/Desktop/Desktop/project/tom_agent_collab
git clone https://github.com/facebookresearch/ToMi.git external/ToMi
```

- [ ] **Step 2: Generate ToMi dataset**

```bash
cd external/ToMi && python main.py && cd ../..
```

This generates `.txt` and `.trace` files in `external/ToMi/data/`.

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_tomi_loader.py
import pytest
from src.data.tomi_loader import (
    parse_tomi_story,
    extract_perspective_pairs,
    load_tomi_dataset,
)

SAMPLE_STORY = """Logan entered the den.
Avery entered the den.
The lettuce is in the green_crate.
Avery exited the den.
Logan moved the lettuce to the blue_box.
Where will Avery look for the lettuce?"""

SAMPLE_TRACE_LINE = "false_belief\t1\tAvery\tlettuce\tgreen_crate\tblue_box"

def test_parse_tomi_story():
    result = parse_tomi_story(SAMPLE_STORY)
    assert "story" in result
    assert "question" in result
    assert "Avery" in result["question"]

def test_extract_perspective_pairs():
    story_data = {
        "story": SAMPLE_STORY.rsplit("\n", 1)[0],
        "question": "Where will Avery look for the lettuce?",
        "agents": ["Logan", "Avery"],
        "answer": "green_crate",
        "true_location": "blue_box",
        "story_type": "false_belief",
    }
    pairs = extract_perspective_pairs(story_data)
    assert len(pairs) == 2
    assert pairs[0]["perspective"] == "self"
    assert pairs[1]["perspective"] == "partner"

def test_load_tomi_dataset_returns_splits():
    # This test requires generated data — skip if not available
    try:
        train, test = load_tomi_dataset("external/ToMi", test_ratio=0.2, seed=42)
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) > len(test)
    except FileNotFoundError:
        pytest.skip("ToMi data not generated yet")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_tomi_loader.py -v`
Expected: FAIL with "cannot import name 'parse_tomi_story'"

- [ ] **Step 5: Write implementation**

```python
# src/data/tomi_loader.py
import os
import re
from typing import List, Dict, Tuple
import random

def parse_tomi_story(text: str) -> Dict:
    """Parse a ToMi story text into structured data."""
    lines = text.strip().split("\n")
    question = lines[-1]
    story = "\n".join(lines[:-1])

    # Extract agent names (names appear as first word of "X entered/exited" lines)
    agents = []
    for line in lines[:-1]:
        match = re.match(r"^(\w+) (entered|exited)", line)
        if match and match.group(1) not in agents:
            agents.append(match.group(1))

    return {"story": story, "question": question, "agents": agents}

def parse_tomi_trace(trace_line: str) -> Dict:
    """Parse a ToMi trace line for ground truth."""
    parts = trace_line.strip().split("\t")
    if len(parts) >= 6:
        return {
            "story_type": parts[0],
            "order": int(parts[1]),
            "agent": parts[2],
            "object": parts[3],
            "belief_location": parts[4],
            "true_location": parts[5],
        }
    return {}

def extract_perspective_pairs(story_data: Dict) -> List[Dict]:
    """Create self-perspective and partner-perspective training examples."""
    agents = story_data.get("agents", [])
    if len(agents) < 2:
        return []

    pairs = []
    # Self perspective: what does the questioned agent believe?
    pairs.append({
        "perspective": "self",
        "story": story_data["story"],
        "question": f"What does {agents[0]} believe about the location of the object?",
        "answer": story_data.get("answer", story_data.get("belief_location", "")),
        "agent": agents[0],
    })
    # Partner perspective: what does the other agent believe?
    pairs.append({
        "perspective": "partner",
        "story": story_data["story"],
        "question": f"What does {agents[1]} believe about the location of the object?",
        "answer": story_data.get("true_location", ""),
        "agent": agents[1],
    })
    return pairs

def load_tomi_dataset(
    tomi_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Load ToMi data and split into train/test perspective pairs.

    Args:
        tomi_dir: path to cloned ToMi repo (contains data/ subdirectory)
        test_ratio: fraction of data for test split
        seed: random seed for reproducible splitting
    """
    data_dir = os.path.join(tomi_dir, "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"ToMi data directory not found: {data_dir}")

    all_pairs = []

    # Find all .txt files and their matching .trace files
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        txt_path = os.path.join(data_dir, fname)
        trace_path = txt_path.replace(".txt", ".trace")

        with open(txt_path) as f:
            stories_raw = f.read().strip().split("\n\n")
        trace_lines = []
        if os.path.exists(trace_path):
            with open(trace_path) as f:
                trace_lines = f.read().strip().split("\n")

        for i, story_text in enumerate(stories_raw):
            parsed = parse_tomi_story(story_text)
            if i < len(trace_lines):
                trace = parse_tomi_trace(trace_lines[i])
                parsed.update(trace)
            pairs = extract_perspective_pairs(parsed)
            all_pairs.extend(pairs)

    # Split
    random.seed(seed)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * (1 - test_ratio))
    return all_pairs[:split_idx], all_pairs[split_idx:]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_tomi_loader.py -v`
Expected: First 2 tests PASS, third may skip if data not generated

- [ ] **Step 7: Commit**

```bash
git add src/data/tomi_loader.py tests/test_tomi_loader.py
git commit -m "feat: ToMi data loader with perspective pair extraction"
```

---

### Task 7: ExploreToM data loader

**Files:**
- Create: `src/data/exploretom_loader.py`
- Create: `tests/test_exploretom_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_exploretom_loader.py
import pytest
from src.data.exploretom_loader import (
    load_exploretom_dataset,
    filter_by_order,
    extract_perspective_pairs_exploretom,
)

def test_filter_by_order():
    """Filter to 1st and 2nd order questions only."""
    examples = [
        {"nth_order": -1, "question": "factual"},
        {"nth_order": 0, "question": "state tracking"},
        {"nth_order": 1, "question": "1st order belief"},
        {"nth_order": 2, "question": "2nd order belief"},
    ]
    filtered = filter_by_order(examples, orders=[1, 2])
    assert len(filtered) == 2
    assert all(e["nth_order"] in [1, 2] for e in filtered)

def test_extract_perspective_pairs_exploretom():
    example = {
        "infilled_story": "Anne put the apple on the table. Bob left the room. Anne moved the apple to the shelf.",
        "question": "Does Bob know where the apple is?",
        "expected_answer": "no",
        "nth_order": 1,
    }
    pairs = extract_perspective_pairs_exploretom(example)
    assert len(pairs) == 2
    assert pairs[0]["perspective"] in ["self", "partner"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_exploretom_loader.py -v`
Expected: FAIL with "cannot import name 'load_exploretom_dataset'"

- [ ] **Step 3: Write implementation**

```python
# src/data/exploretom_loader.py
from typing import List, Dict, Tuple
import re

def load_exploretom_dataset() -> List[Dict]:
    """Load ExploreToM from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("facebook/ExploreToM", split="test")
    return [dict(row) for row in ds]

def filter_by_order(examples: List[Dict], orders: List[int] = [1, 2]) -> List[Dict]:
    """Filter to specific ToM order questions."""
    return [e for e in examples if e.get("nth_order") in orders]

def extract_agents_from_story(story: str) -> List[str]:
    """Extract agent names from an ExploreToM story."""
    # Common patterns: "X put", "X moved", "X left", "X entered"
    agents = []
    for match in re.finditer(r"\b([A-Z][a-z]+)\b(?= (?:put|moved|left|entered|took|went|is|was|said|told|know|think))", story):
        name = match.group(1)
        if name not in agents:
            agents.append(name)
    return agents

def extract_perspective_pairs_exploretom(example: Dict) -> List[Dict]:
    """Create perspective pairs from an ExploreToM example."""
    story = example.get("infilled_story", "")
    question = example.get("question", "")
    answer = example.get("expected_answer", "")
    agents = extract_agents_from_story(story)

    if len(agents) < 2:
        # Fallback: extract from question
        q_agents = re.findall(r"\b([A-Z][a-z]+)\b", question)
        for a in q_agents:
            if a not in agents:
                agents.append(a)

    if len(agents) < 2:
        return []

    pairs = [
        {
            "perspective": "self",
            "story": story,
            "question": f"From {agents[0]}'s perspective: {question}",
            "answer": answer,
            "agent": agents[0],
            "nth_order": example.get("nth_order", 1),
        },
        {
            "perspective": "partner",
            "story": story,
            "question": f"From {agents[1]}'s perspective: {question}",
            "answer": answer,
            "agent": agents[1],
            "nth_order": example.get("nth_order", 1),
        },
    ]
    return pairs

def load_and_prepare_exploretom(
    orders: List[int] = [1, 2],
) -> List[Dict]:
    """Load ExploreToM, filter by order, extract perspective pairs."""
    raw = load_exploretom_dataset()
    filtered = filter_by_order(raw, orders=orders)
    all_pairs = []
    for example in filtered:
        pairs = extract_perspective_pairs_exploretom(example)
        all_pairs.extend(pairs)
    return all_pairs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_exploretom_loader.py -v`
Expected: All tests PASS (first 2 tests use mock data, no HF download needed)

- [ ] **Step 5: Commit**

```bash
git add src/data/exploretom_loader.py tests/test_exploretom_loader.py
git commit -m "feat: ExploreToM data loader with order filtering and perspective pairs"
```

---

### Task 8: Combined training data pipeline

**Files:**
- Create: `src/data/prefix_training_data.py`

- [ ] **Step 1: Write implementation**

```python
# src/data/prefix_training_data.py
from typing import List, Dict, Tuple
import random
from collections import Counter

from src.data.tomi_loader import load_tomi_dataset
from src.data.exploretom_loader import load_and_prepare_exploretom

def validate_dataset(data: List[Dict], source_name: str) -> None:
    """Run data validation checks and print statistics."""
    print(f"\n=== {source_name} Dataset Validation ===")
    print(f"Total examples: {len(data)}")

    # Label balance
    perspectives = Counter(d["perspective"] for d in data)
    print(f"Perspective distribution: {dict(perspectives)}")

    # Sequence length distribution (approximate by character count)
    lengths = [len(d.get("story", "") + d.get("question", "")) for d in data]
    if lengths:
        lengths_sorted = sorted(lengths)
        print(f"Char lengths — mean: {sum(lengths)/len(lengths):.0f}, "
              f"p50: {lengths_sorted[len(lengths)//2]}, "
              f"p95: {lengths_sorted[int(len(lengths)*0.95)]}, "
              f"max: {lengths_sorted[-1]}")

    # Spot check
    if len(data) >= 10:
        samples = random.sample(data, 10)
        print("Spot check (10 random examples):")
        for s in samples[:3]:
            print(f"  [{s['perspective']}] Q: {s['question'][:80]}... A: {s.get('answer', 'N/A')[:40]}")

def combine_training_data(
    tomi_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Combine ToMi + ExploreToM into training and test sets.

    Returns:
        (train_data, test_data) — test_data is ToMi-only (for ToMi eval)
    """
    # Load ToMi with train/test split
    tomi_train, tomi_test = load_tomi_dataset(tomi_dir, test_ratio=test_ratio, seed=seed)
    validate_dataset(tomi_train, "ToMi (train)")
    validate_dataset(tomi_test, "ToMi (test)")

    # Load ExploreToM (all goes to training — separate from eval benchmarks)
    exploretom = load_and_prepare_exploretom(orders=[1, 2])
    validate_dataset(exploretom, "ExploreToM")

    # Combine for training
    train_data = tomi_train + exploretom
    random.seed(seed)
    random.shuffle(train_data)

    print(f"\n=== Combined Training Set ===")
    print(f"Total: {len(train_data)} (ToMi: {len(tomi_train)}, ExploreToM: {len(exploretom)})")

    return train_data, tomi_test

def format_for_prefix_training(
    example: Dict,
    prefix_type: str,
) -> str:
    """Format a single example for prefix tuning.

    Args:
        example: a perspective pair dict
        prefix_type: "self" or "partner" — only include examples matching this prefix
    """
    if example["perspective"] != prefix_type:
        return None

    return (
        f"Story: {example['story']}\n"
        f"Question: {example['question']}\n"
        f"Answer: {example.get('answer', '')}"
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/data/prefix_training_data.py
git commit -m "feat: combined training data pipeline with validation"
```

---

### Task 9: SimpleToM evaluator

**Files:**
- Create: `src/evaluation/simpletom_eval.py`

- [ ] **Step 1: Write implementation**

```python
# src/evaluation/simpletom_eval.py
from typing import List, Dict, Callable, Tuple
from datasets import load_dataset
import numpy as np

def load_simpletom() -> List[Dict]:
    """Load SimpleToM dataset from HuggingFace."""
    ds = load_dataset("allenai/SimpleToM", split="test")
    return [dict(row) for row in ds]

def categorize_questions(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Split SimpleToM into mental_state / behavior / judgment categories."""
    categories = {"mental_state": [], "behavior": [], "judgment": []}
    for item in data:
        qtype = item.get("question_type", "").lower()
        if "mental" in qtype or "aware" in qtype or "know" in qtype:
            categories["mental_state"].append(item)
        elif "behavior" in qtype or "action" in qtype or "will" in qtype:
            categories["behavior"].append(item)
        elif "judg" in qtype or "reasonable" in qtype:
            categories["judgment"].append(item)
        else:
            # Try to categorize by question order (a=mental, b=behavior, c=judgment)
            q_order = item.get("question_order", "")
            if q_order == "a":
                categories["mental_state"].append(item)
            elif q_order == "b":
                categories["behavior"].append(item)
            elif q_order == "c":
                categories["judgment"].append(item)
    return categories

def evaluate_simpletom(
    data: List[Dict],
    predict_fn: Callable[[str, str], Tuple[str, float]],
) -> Dict[str, float]:
    """Evaluate a prediction function on SimpleToM.

    Args:
        data: SimpleToM dataset
        predict_fn: (story, question) → (predicted_answer, confidence)

    Returns:
        Dict with accuracy per category and overall
    """
    categories = categorize_questions(data)
    results = {}

    for cat_name, cat_data in categories.items():
        if not cat_data:
            continue
        correct = 0
        confidences = []
        correctness = []
        for item in cat_data:
            story = item.get("story", item.get("narrative", ""))
            question = item.get("question", "")
            expected = item.get("answer", item.get("expected_answer", "")).strip().lower()

            pred, conf = predict_fn(story, question)
            pred = pred.strip().lower()
            is_correct = pred == expected or pred in expected or expected in pred
            correct += int(is_correct)
            confidences.append(conf)
            correctness.append(int(is_correct))

        acc = correct / len(cat_data) if cat_data else 0.0
        results[f"{cat_name}_accuracy"] = acc
        results[f"{cat_name}_count"] = len(cat_data)
        results[f"{cat_name}_confidences"] = np.array(confidences)
        results[f"{cat_name}_correctness"] = np.array(correctness)

    total_correct = sum(results.get(f"{c}_accuracy", 0) * results.get(f"{c}_count", 0)
                        for c in categories)
    total_count = sum(results.get(f"{c}_count", 0) for c in categories)
    results["overall_accuracy"] = total_correct / total_count if total_count > 0 else 0.0

    return results
```

- [ ] **Step 2: Commit**

```bash
git add src/evaluation/simpletom_eval.py
git commit -m "feat: SimpleToM evaluator with per-category accuracy"
```

---

### Task 10: CoordinationQA evaluator

**Files:**
- Create: `src/evaluation/coordination_qa.py`

- [ ] **Step 1: Clone LLM-Coordination repository**

```bash
cd /Users/quyetthang/Desktop/Desktop/project/tom_agent_collab
git clone https://github.com/eric-ai-lab/llm_coordination.git external/llm_coordination
```

- [ ] **Step 2: Write implementation**

```python
# src/evaluation/coordination_qa.py
import json
import os
from typing import List, Dict, Callable, Tuple
import numpy as np

def load_coordination_qa(repo_dir: str) -> List[Dict]:
    """Load CoordinationQA questions from cloned llm_coordination repo.

    Searches for JSON/CSV files containing the 198 MC questions.
    """
    # The repo structure may vary — search for QA data files
    qa_data = []
    for root, dirs, files in os.walk(repo_dir):
        for fname in files:
            if "coordinationqa" in fname.lower() or "qa" in fname.lower():
                fpath = os.path.join(root, fname)
                if fname.endswith(".json"):
                    with open(fpath) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        qa_data.extend(data)
                    elif isinstance(data, dict):
                        for v in data.values():
                            if isinstance(v, list):
                                qa_data.extend(v)
    return qa_data

def categorize_coordination_qa(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Split CoordinationQA into env_comprehension / tom / joint_planning."""
    categories = {"env_comprehension": [], "tom_reasoning": [], "joint_planning": []}
    for item in data:
        dim = item.get("dimension", item.get("category", "")).lower()
        if "env" in dim or "comprehension" in dim:
            categories["env_comprehension"].append(item)
        elif "tom" in dim or "theory" in dim or "mind" in dim:
            categories["tom_reasoning"].append(item)
        elif "plan" in dim or "joint" in dim:
            categories["joint_planning"].append(item)
    return categories

def evaluate_coordination_qa(
    data: List[Dict],
    predict_fn: Callable[[str, List[str]], Tuple[int, float]],
) -> Dict[str, float]:
    """Evaluate on CoordinationQA.

    Args:
        data: list of MC question dicts
        predict_fn: (question_text, options) → (chosen_index, confidence)

    Returns:
        Dict with accuracy per dimension and overall
    """
    categories = categorize_coordination_qa(data)
    results = {}

    for cat_name, cat_data in categories.items():
        if not cat_data:
            continue
        correct = 0
        for item in cat_data:
            question = item.get("question", "")
            options = item.get("options", item.get("choices", []))
            correct_idx = item.get("answer_index", item.get("correct", 0))
            if isinstance(correct_idx, str):
                correct_idx = ord(correct_idx.upper()) - ord("A")

            pred_idx, conf = predict_fn(question, options)
            correct += int(pred_idx == correct_idx)

        acc = correct / len(cat_data) if cat_data else 0.0
        results[f"{cat_name}_accuracy"] = acc
        results[f"{cat_name}_count"] = len(cat_data)

    total_correct = sum(results.get(f"{c}_accuracy", 0) * results.get(f"{c}_count", 0)
                        for c in categories)
    total_count = sum(results.get(f"{c}_count", 0) for c in categories)
    results["overall_accuracy"] = total_correct / total_count if total_count > 0 else 0.0

    return results
```

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/coordination_qa.py
git commit -m "feat: CoordinationQA evaluator with per-dimension accuracy"
```

---

### Task 11: ToMi evaluator

**Files:**
- Create: `src/evaluation/tomi_eval.py`

- [ ] **Step 1: Write implementation**

```python
# src/evaluation/tomi_eval.py
from typing import List, Dict, Callable, Tuple
import numpy as np

def evaluate_tomi(
    test_data: List[Dict],
    predict_fn: Callable[[str, str], Tuple[str, float]],
) -> Dict[str, float]:
    """Evaluate on ToMi test split.

    Args:
        test_data: ToMi test split (perspective pairs with ground truth)
        predict_fn: (story, question) → (predicted_answer, confidence)

    Returns:
        Dict with accuracy per story_type and overall
    """
    by_type = {}
    for item in test_data:
        stype = item.get("story_type", "unknown")
        if stype not in by_type:
            by_type[stype] = []
        by_type[stype].append(item)

    results = {}
    all_correct = 0
    all_total = 0

    for stype, items in by_type.items():
        correct = 0
        confidences = []
        correctness = []
        for item in items:
            story = item.get("story", "")
            question = item.get("question", "")
            expected = item.get("answer", item.get("belief_location", "")).strip().lower()

            pred, conf = predict_fn(story, question)
            pred = pred.strip().lower()
            is_correct = pred == expected or expected in pred
            correct += int(is_correct)
            confidences.append(conf)
            correctness.append(int(is_correct))

        acc = correct / len(items) if items else 0.0
        results[f"{stype}_accuracy"] = acc
        results[f"{stype}_count"] = len(items)
        results[f"{stype}_confidences"] = np.array(confidences)
        results[f"{stype}_correctness"] = np.array(correctness)
        all_correct += correct
        all_total += len(items)

    results["overall_accuracy"] = all_correct / all_total if all_total > 0 else 0.0
    return results
```

- [ ] **Step 2: Commit**

```bash
git add src/evaluation/tomi_eval.py
git commit -m "feat: ToMi evaluator with per-story-type accuracy"
```

---

## Chunk 3: Model Loading, Baselines, and Perspective Runner

### Task 12: Model loader

**Files:**
- Create: `src/models/model_loader.py`

- [ ] **Step 1: Write implementation**

```python
# src/models/model_loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import PAECConfig

def load_base_model(config: PAECConfig):
    """Load Qwen3-1.7B in fp16 with optional gradient checkpointing."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer

def load_model_with_prefix(config: PAECConfig, prefix_path: str):
    """Load base model with a saved prefix checkpoint."""
    model, tokenizer = load_base_model(config)
    model = PeftModel.from_pretrained(model, prefix_path)
    model.eval()
    return model, tokenizer

def get_answer_token_ids(tokenizer, options: list) -> list:
    """Get token IDs for answer options (A, B, C, D or yes/no)."""
    token_ids = []
    for opt in options:
        ids = tokenizer.encode(opt, add_special_tokens=False)
        token_ids.append(ids[0] if ids else tokenizer.unk_token_id)
    return token_ids
```

- [ ] **Step 2: Commit**

```bash
git add src/models/model_loader.py
git commit -m "feat: model loader with prefix support and answer token extraction"
```

---

### Task 13: Baselines

**Files:**
- Create: `src/baselines/standard_prompting.py`
- Create: `src/baselines/simtom_prompting.py`
- Create: `src/baselines/self_consistency.py`
- Create: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baselines.py
import pytest

def test_format_standard_prompt():
    from src.baselines.standard_prompting import format_prompt
    prompt = format_prompt("A story here.", "What does X believe?", ["Room A", "Room B"])
    assert "A story here" in prompt
    assert "Room A" in prompt
    assert "Room B" in prompt

def test_format_simtom_prompt():
    from src.baselines.simtom_prompting import format_simtom_prompt
    prompt = format_simtom_prompt("A story.", "What does Alice believe?", ["A", "B"], agent="Alice")
    assert "Alice" in prompt
    assert "perspective" in prompt.lower() or "viewpoint" in prompt.lower() or "only" in prompt.lower()

def test_format_sc_prompts():
    from src.baselines.self_consistency import format_sc_prompts
    prompts = format_sc_prompts("A story.", "What?", ["A", "B"], n_samples=4)
    assert len(prompts) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: FAIL

- [ ] **Step 3: Write standard prompting baseline**

```python
# src/baselines/standard_prompting.py
import torch
from typing import List, Tuple

def format_prompt(story: str, question: str, options: List[str]) -> str:
    """Format a standard MC prompt."""
    options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    return (
        f"Read the following story and answer the question.\n\n"
        f"Story: {story}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with just the letter (A, B, etc.):"
    )

def predict_standard(model, tokenizer, story: str, question: str, options: List[str]) -> Tuple[int, float]:
    """Single-pass prediction with argmax on answer token logits."""
    prompt = format_prompt(story, question, options)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits at the last position
    last_logits = outputs.logits[0, -1, :]

    # Get logits for answer tokens (A, B, C, D)
    answer_letters = [chr(65 + i) for i in range(len(options))]
    answer_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in answer_letters]
    answer_logits = last_logits[answer_ids]

    probs = torch.softmax(answer_logits, dim=0)
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    return pred_idx, confidence
```

- [ ] **Step 4: Write SimToM baseline**

```python
# src/baselines/simtom_prompting.py
import torch
from typing import List, Tuple

def format_simtom_prompt(story: str, question: str, options: List[str], agent: str = "") -> str:
    """Format a SimToM perspective-taking prompt."""
    options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    perspective_instruction = (
        f"Consider only what {agent} has directly observed or been told. "
        f"Ignore any information that {agent} would not have access to. "
        f"Think step by step about what {agent} knows and doesn't know."
    ) if agent else (
        "Consider only what the relevant agent has directly observed. "
        "Think step by step about what they know and don't know."
    )
    return (
        f"{perspective_instruction}\n\n"
        f"Story: {story}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with just the letter (A, B, etc.):"
    )

def predict_simtom(model, tokenizer, story: str, question: str, options: List[str], agent: str = "") -> Tuple[int, float]:
    """SimToM prediction: single pass with perspective prompt."""
    prompt = format_simtom_prompt(story, question, options, agent=agent)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_logits = outputs.logits[0, -1, :]
    answer_letters = [chr(65 + i) for i in range(len(options))]
    answer_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in answer_letters]
    answer_logits = last_logits[answer_ids]

    probs = torch.softmax(answer_logits, dim=0)
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    return pred_idx, confidence
```

- [ ] **Step 5: Write self-consistency baseline**

```python
# src/baselines/self_consistency.py
import torch
from typing import List, Tuple
from collections import Counter

def format_sc_prompts(story: str, question: str, options: List[str], n_samples: int = 2) -> List[str]:
    """Generate n_samples prompts for self-consistency voting."""
    options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    base_prompt = (
        f"Read the following story and answer the question.\n\n"
        f"Story: {story}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Think step by step, then answer with just the letter (A, B, etc.):"
    )
    return [base_prompt] * n_samples

def predict_self_consistency(
    model,
    tokenizer,
    story: str,
    question: str,
    options: List[str],
    n_samples: int = 2,
    temperature: float = 0.7,
) -> Tuple[int, float]:
    """Self-consistency: sample N times with temperature, majority vote."""
    prompts = format_sc_prompts(story, question, options, n_samples)
    votes = []

    answer_letters = [chr(65 + i) for i in range(len(options))]
    answer_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in answer_letters]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_logits = outputs.logits[0, -1, :]
        answer_logits = last_logits[answer_ids]

        # Sample with temperature
        if temperature > 0:
            probs = torch.softmax(answer_logits / temperature, dim=0)
            pred_idx = torch.multinomial(probs, 1).item()
        else:
            pred_idx = answer_logits.argmax().item()

        votes.append(pred_idx)

    # Majority vote
    vote_counts = Counter(votes)
    winner = vote_counts.most_common(1)[0]
    pred_idx = winner[0]
    confidence = winner[1] / n_samples

    return pred_idx, confidence
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: All 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/baselines/ tests/test_baselines.py
git commit -m "feat: baselines — standard prompting, SimToM, self-consistency"
```

---

### Task 14: Perspective runner (PAEC inference orchestrator)

**Files:**
- Create: `src/inference/perspective_runner.py`
- Create: `tests/test_perspective_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_perspective_runner.py
import pytest
import torch
from unittest.mock import MagicMock
from src.inference.perspective_runner import PerspectiveRunner
from src.config import PAECConfig

def test_runner_returns_fused_result():
    """PerspectiveRunner produces a FusedResult with valid structure."""
    # Mock model that returns random logits
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 10, 151936)  # (batch, seq, vocab)
    mock_model.return_value = mock_output
    mock_model.device = torch.device("cpu")

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long), "attention_mask": torch.ones(1, 10, dtype=torch.long)}
    mock_tokenizer.encode = lambda x, add_special_tokens=False: [65]  # dummy token id
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    config = PAECConfig()
    runner = PerspectiveRunner(config)

    result = runner.predict(
        model_self=mock_model,
        model_partner=mock_model,
        tokenizer=mock_tokenizer,
        story="A story",
        question="What does X believe?",
        options=["Room A", "Room B"],
    )

    assert hasattr(result, "answer_idx")
    assert hasattr(result, "fused")
    assert result.answer_idx in [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_perspective_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/inference/perspective_runner.py
from dataclasses import dataclass
from typing import List, Tuple
import torch

from src.config import PAECConfig
from src.inference.logit_to_dirichlet import logits_to_opinion, Opinion
from src.inference.ds_fusion import dempster_combine, FusedResult

@dataclass
class PAECResult:
    answer_idx: int
    answer_text: str
    fused: FusedResult
    opinion_self: Opinion
    opinion_partner: Opinion
    raw_logits_self: torch.Tensor
    raw_logits_partner: torch.Tensor
    used_fallback: bool

class PerspectiveRunner:
    def __init__(self, config: PAECConfig):
        self.config = config

    def _get_answer_logits(
        self,
        model,
        tokenizer,
        prompt: str,
        options: List[str],
    ) -> torch.Tensor:
        """Run a single forward pass and extract logits for answer tokens."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_seq_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_logits = outputs.logits[0, -1, :]

        # Get logits for answer option tokens
        answer_letters = [chr(65 + i) for i in range(len(options))]
        answer_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in answer_letters]
        return last_logits[answer_ids]

    def _format_perspective_prompt(
        self,
        story: str,
        question: str,
        options: List[str],
        perspective: str,
    ) -> str:
        """Format prompt with perspective framing (used for prompt-only mode)."""
        options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

        if perspective == "self":
            prefix = "Reason from the viewpoint of the agent being asked about. Consider only what they have directly seen or been told."
        else:
            prefix = "Reason from the viewpoint of the other agent. Consider only what they have directly seen or been told."

        return (
            f"{prefix}\n\n"
            f"Story: {story}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Answer:"
        )

    def predict(
        self,
        model_self,
        model_partner,
        tokenizer,
        story: str,
        question: str,
        options: List[str],
        use_prompt_perspective: bool = False,
    ) -> PAECResult:
        """Run PAEC 2-pass inference and return fused result.

        Args:
            model_self: model with self-prefix (or base model for prompt-only)
            model_partner: model with partner-prefix (or base model for prompt-only)
            tokenizer: shared tokenizer
            story, question, options: the input scenario
            use_prompt_perspective: if True, use perspective prompts instead of prefixes
        """
        if use_prompt_perspective:
            prompt_self = self._format_perspective_prompt(story, question, options, "self")
            prompt_partner = self._format_perspective_prompt(story, question, options, "partner")
        else:
            # With prefix models, use a neutral prompt (prefix does the steering)
            options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            prompt = f"Story: {story}\n\nQuestion: {question}\n\nOptions:\n{options_str}\n\nAnswer:"
            prompt_self = prompt
            prompt_partner = prompt

        # Pass 1: self-perspective
        logits_self = self._get_answer_logits(model_self, tokenizer, prompt_self, options)
        opinion_self = logits_to_opinion(
            logits_self,
            logit_transform=self.config.logit_transform,
            evidence_fn=self.config.evidence_fn,
        )

        # Pass 2: partner-perspective
        logits_partner = self._get_answer_logits(model_partner, tokenizer, prompt_partner, options)
        opinion_partner = logits_to_opinion(
            logits_partner,
            logit_transform=self.config.logit_transform,
            evidence_fn=self.config.evidence_fn,
        )

        # Fuse
        fused = dempster_combine(opinion_self, opinion_partner, conflict_epsilon=self.config.conflict_epsilon)

        # Answer selection with degenerate fallback
        used_fallback = False
        if fused.vacuity > self.config.degenerate_vacuity_threshold:
            answer_idx = logits_self.argmax().item()
            used_fallback = True
        else:
            answer_idx = fused.projected_prob.argmax().item()

        return PAECResult(
            answer_idx=answer_idx,
            answer_text=chr(65 + answer_idx),
            fused=fused,
            opinion_self=opinion_self,
            opinion_partner=opinion_partner,
            raw_logits_self=logits_self,
            raw_logits_partner=logits_partner,
            used_fallback=used_fallback,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_perspective_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/inference/perspective_runner.py tests/test_perspective_runner.py
git commit -m "feat: perspective runner — 2-pass PAEC inference with DS fusion"
```

---

## Chunk 4: Prefix Training and Colab Notebooks

### Task 15: Perspective prefix trainer

**Files:**
- Create: `src/models/perspective_prefix.py`

- [ ] **Step 1: Write implementation**

```python
# src/models/perspective_prefix.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import PrefixTuningConfig, get_peft_model, TaskType
from tqdm import tqdm

from src.config import PAECConfig
from src.models.model_loader import load_base_model

class PerspectiveDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = f"Story: {ex['story']}\nQuestion: {ex['question']}\nAnswer: {ex.get('answer', '')}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        # Mask loss on everything except answer tokens
        answer_start = text.find("Answer:") + len("Answer: ")
        answer_token_start = len(self.tokenizer.encode(text[:answer_start], add_special_tokens=False))
        labels[:answer_token_start] = -100
        # Mask padding
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def create_prefix_model(config: PAECConfig, base_model):
    """Wrap base model with prefix tuning."""
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=config.num_virtual_tokens,
        prefix_projection=config.prefix_projection,
        encoder_hidden_size=config.encoder_hidden_size,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model

def train_prefix(
    config: PAECConfig,
    train_data: list,
    perspective: str,
    output_dir: str,
    tokenizer=None,
    base_model=None,
):
    """Train a single perspective prefix.

    Args:
        config: PAEC config
        train_data: list of perspective-labeled examples
        perspective: "self" or "partner" — filter to this perspective
        output_dir: where to save the trained prefix
        tokenizer: pre-loaded tokenizer (optional, loads if None)
        base_model: pre-loaded base model (optional, loads if None)
    """
    if base_model is None or tokenizer is None:
        base_model, tokenizer = load_base_model(config)

    # Filter to matching perspective
    filtered = [ex for ex in train_data if ex["perspective"] == perspective]
    print(f"Training {perspective} prefix on {len(filtered)} examples")

    # Create prefix model
    model = create_prefix_model(config, base_model)

    # Dataset and DataLoader
    dataset = PerspectiveDataset(filtered, tokenizer, max_length=config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Optimizer
    if config.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)
        except ImportError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Scheduler
    total_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    model.train()
    global_step = 0
    for epoch in range(config.num_epochs):
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        # Checkpoint after each epoch
        if config.checkpoint_every_epoch:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            model.save_pretrained(epoch_dir)
            print(f"Saved checkpoint to {epoch_dir}")

    # Save final
    model.save_pretrained(output_dir)
    print(f"Saved final {perspective} prefix to {output_dir}")

    return model
```

- [ ] **Step 2: Commit**

```bash
git add src/models/perspective_prefix.py
git commit -m "feat: perspective prefix trainer with checkpointing"
```

---

### Task 16: Colab notebook — prefix training

**Files:**
- Create: `notebooks/01_prefix_training.ipynb`

- [ ] **Step 1: Write the notebook**

Create a Jupyter notebook with these cells:

**Cell 1 — Setup:**
```python
# Install dependencies
!pip install -q torch transformers peft datasets bitsandbytes accelerate tqdm pyyaml scipy scikit-learn

# Mount Google Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Clone the project
!git clone https://github.com/<your-repo>/tom_agent_collab.git /content/paec
%cd /content/paec

# Clone external repos
!git clone https://github.com/facebookresearch/ToMi.git external/ToMi
```

**Cell 2 — Generate ToMi data:**
```python
%cd external/ToMi
!python main.py
%cd /content/paec
```

**Cell 3 — Load config and data:**
```python
from src.config import PAECConfig
from src.data.prefix_training_data import combine_training_data

config = PAECConfig.from_yaml("configs/default.yaml")
train_data, tomi_test = combine_training_data("external/ToMi", seed=42)
print(f"Training examples: {len(train_data)}")
print(f"ToMi test examples: {len(tomi_test)}")
```

**Cell 4 — Train self-perspective prefix:**
```python
from src.models.model_loader import load_base_model
from src.models.perspective_prefix import train_prefix

base_model, tokenizer = load_base_model(config)

self_prefix_model = train_prefix(
    config=config,
    train_data=train_data,
    perspective="self",
    output_dir="/content/drive/MyDrive/paec_checkpoints/prefix_self",
    tokenizer=tokenizer,
    base_model=base_model,
)
```

**Cell 5 — Train partner-perspective prefix:**
```python
# Reload base model (prefix training modifies it in-place)
base_model, tokenizer = load_base_model(config)

partner_prefix_model = train_prefix(
    config=config,
    train_data=train_data,
    perspective="partner",
    output_dir="/content/drive/MyDrive/paec_checkpoints/prefix_partner",
    tokenizer=tokenizer,
    base_model=base_model,
)
```

**Cell 6 — Verify prefix diversity:**
```python
import torch
from peft import PeftModel

# Load both prefixes and compare embeddings
model1, _ = load_base_model(config)
model1 = PeftModel.from_pretrained(model1, "/content/drive/MyDrive/paec_checkpoints/prefix_self")

model2, _ = load_base_model(config)
model2 = PeftModel.from_pretrained(model2, "/content/drive/MyDrive/paec_checkpoints/prefix_partner")

# Extract prefix embeddings and compute cosine similarity
emb1 = model1.get_prompt(batch_size=1).detach().flatten()
emb2 = model2.get_prompt(batch_size=1).detach().flatten()

cosine_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
print(f"Prefix cosine similarity: {cosine_sim.item():.4f}")
print("(Lower is better — indicates distinct perspectives)")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_prefix_training.ipynb
git commit -m "feat: Colab notebook for prefix training on T4"
```

---

### Task 17: Colab notebook — inference pipeline

**Files:**
- Create: `notebooks/02_inference_pipeline.ipynb`

- [ ] **Step 1: Write the notebook**

Create a Jupyter notebook with cells for:

**Cell 1 — Setup (same as 01)**

**Cell 2 — Load models:**
```python
from src.config import PAECConfig
from src.models.model_loader import load_base_model, load_model_with_prefix
from src.inference.perspective_runner import PerspectiveRunner

config = PAECConfig.from_yaml("configs/default.yaml")

# Load prefix models
model_self, tokenizer = load_model_with_prefix(config, "/content/drive/MyDrive/paec_checkpoints/prefix_self")
model_partner, _ = load_model_with_prefix(config, "/content/drive/MyDrive/paec_checkpoints/prefix_partner")

runner = PerspectiveRunner(config)
```

**Cell 3 — Quick test with a sample story:**
```python
result = runner.predict(
    model_self=model_self,
    model_partner=model_partner,
    tokenizer=tokenizer,
    story="Alice put the ball in the basket. Alice left the room. Bob moved the ball to the box.",
    question="Where will Alice look for the ball?",
    options=["basket", "box"],
    use_prompt_perspective=False,
)

print(f"Answer: {result.answer_text} (idx={result.answer_idx})")
print(f"Vacuity: {result.fused.vacuity:.4f}")
print(f"Dissonance: {result.fused.normalized_conflict:.4f}")
print(f"Confidence: {result.fused.confidence:.4f}")
print(f"Used fallback: {result.used_fallback}")
print(f"Self opinion:    belief={result.opinion_self.belief}, vacuity={result.opinion_self.vacuity:.4f}")
print(f"Partner opinion: belief={result.opinion_partner.belief}, vacuity={result.opinion_partner.vacuity:.4f}")
```

**Cell 4 — Run prompt-only baseline (Phase 1 / ablation):**
```python
base_model, tokenizer = load_base_model(config)
result_prompt = runner.predict(
    model_self=base_model,
    model_partner=base_model,
    tokenizer=tokenizer,
    story="Alice put the ball in the basket. Alice left the room. Bob moved the ball to the box.",
    question="Where will Alice look for the ball?",
    options=["basket", "box"],
    use_prompt_perspective=True,
)
print(f"Prompt-only answer: {result_prompt.answer_text}, vacuity: {result_prompt.fused.vacuity:.4f}")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/02_inference_pipeline.ipynb
git commit -m "feat: Colab notebook for PAEC inference pipeline"
```

---

### Task 18: Colab notebook — full evaluation

**Files:**
- Create: `notebooks/03_evaluation.ipynb`

- [ ] **Step 1: Write the notebook**

Notebook cells covering:

**Cell 1 — Setup + load models (same as 02)**

**Cell 2 — Run all methods on SimpleToM:**
```python
from src.evaluation.simpletom_eval import load_simpletom, evaluate_simpletom
from src.baselines.standard_prompting import predict_standard
from src.baselines.simtom_prompting import predict_simtom
from src.baselines.self_consistency import predict_self_consistency

simpletom_data = load_simpletom()
print(f"SimpleToM: {len(simpletom_data)} examples")

# Baseline 1: Standard prompting
results_standard = evaluate_simpletom(simpletom_data, lambda s, q: predict_standard(base_model, tokenizer, s, q, ["yes", "no"]))
print(f"Standard: {results_standard}")

# Baseline 2: SimToM
results_simtom = evaluate_simpletom(simpletom_data, lambda s, q: predict_simtom(base_model, tokenizer, s, q, ["yes", "no"]))
print(f"SimToM: {results_simtom}")

# PAEC (prompt-only — Phase 1)
def paec_prompt_predict(story, question):
    r = runner.predict(base_model, base_model, tokenizer, story, question, ["yes", "no"], use_prompt_perspective=True)
    return ["yes", "no"][r.answer_idx], r.fused.confidence
results_paec_prompt = evaluate_simpletom(simpletom_data, paec_prompt_predict)
print(f"PAEC (prompt-only): {results_paec_prompt}")

# PAEC (prefix — Phase 2)
def paec_prefix_predict(story, question):
    r = runner.predict(model_self, model_partner, tokenizer, story, question, ["yes", "no"])
    return ["yes", "no"][r.answer_idx], r.fused.confidence
results_paec_prefix = evaluate_simpletom(simpletom_data, paec_prefix_predict)
print(f"PAEC (prefix): {results_paec_prefix}")
```

**Cell 3 — Run on ToMi test split:**
```python
from src.evaluation.tomi_eval import evaluate_tomi
from src.data.prefix_training_data import combine_training_data

_, tomi_test = combine_training_data("external/ToMi", seed=42)

# (similar pattern: run all methods, collect results)
```

**Cell 4 — Run on CoordinationQA:**
```python
from src.evaluation.coordination_qa import load_coordination_qa, evaluate_coordination_qa

cqa_data = load_coordination_qa("external/llm_coordination")
print(f"CoordinationQA: {len(cqa_data)} questions")

# (similar pattern: run all methods, collect results)
```

**Cell 5 — Calibration analysis:**
```python
from src.evaluation.calibration import expected_calibration_error, brier_score, bootstrap_ci
import numpy as np

# Collect confidences and correctness from PAEC runs
# Compute ECE, Brier, bootstrap CIs
# Compare against SC baselines
```

**Cell 6 — Results summary table:**
```python
import pandas as pd

# Aggregate all results into a comparison table
# Print formatted table for paper
```

**Cell 7 — Vacuity / dissonance analysis:**
```python
# Analyze high-vacuity and high-dissonance cases
# Plot accuracy-when-confident curves
# Select case studies for paper
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/03_evaluation.ipynb
git commit -m "feat: Colab notebook for full evaluation suite"
```

---

## Chunk 5: Experiment Orchestrator and Final Integration

### Task 19: Run-all-experiments script

**Files:**
- Create: `scripts/run_all_experiments.py`

- [ ] **Step 1: Write implementation**

```python
# scripts/run_all_experiments.py
"""Orchestrate all PAEC experiments.

Usage:
    python scripts/run_all_experiments.py --config configs/default.yaml --phase 1
    python scripts/run_all_experiments.py --config configs/default.yaml --phase 2
    python scripts/run_all_experiments.py --config configs/default.yaml --all
"""
import argparse
import json
import os
import torch
import numpy as np
from datetime import datetime

from src.config import PAECConfig
from src.models.model_loader import load_base_model, load_model_with_prefix
from src.inference.perspective_runner import PerspectiveRunner
from src.baselines.standard_prompting import predict_standard
from src.baselines.simtom_prompting import predict_simtom
from src.baselines.self_consistency import predict_self_consistency
from src.evaluation.simpletom_eval import load_simpletom, evaluate_simpletom
from src.evaluation.tomi_eval import evaluate_tomi
from src.evaluation.coordination_qa import load_coordination_qa, evaluate_coordination_qa
from src.evaluation.calibration import expected_calibration_error, brier_score, bootstrap_ci

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_experiment(config: PAECConfig, phase: int, output_dir: str):
    """Run experiments for a given phase."""
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for seed in config.seeds:
        set_seed(seed)
        print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")

        # Load models
        base_model, tokenizer = load_base_model(config)
        runner = PerspectiveRunner(config)

        seed_results = {}

        # Phase 1: prompt-only baselines
        if phase >= 1:
            print("\n--- Phase 1: Baselines + Prompt-only PAEC ---")

            # Load SimpleToM
            simpletom = load_simpletom()

            # Standard prompting
            def std_predict(s, q):
                idx, conf = predict_standard(base_model, tokenizer, s, q, ["yes", "no"])
                return ["yes", "no"][idx], conf
            seed_results["standard_simpletom"] = evaluate_simpletom(simpletom, std_predict)

            # SimToM
            def sim_predict(s, q):
                idx, conf = predict_simtom(base_model, tokenizer, s, q, ["yes", "no"])
                return ["yes", "no"][idx], conf
            seed_results["simtom_simpletom"] = evaluate_simpletom(simpletom, sim_predict)

            # Self-consistency N=2
            def sc2_predict(s, q):
                idx, conf = predict_self_consistency(base_model, tokenizer, s, q, ["yes", "no"], n_samples=2, temperature=config.sc_temperature)
                return ["yes", "no"][idx], conf
            seed_results["sc2_simpletom"] = evaluate_simpletom(simpletom, sc2_predict)

            # PAEC prompt-only
            def paec_prompt_predict(s, q):
                r = runner.predict(base_model, base_model, tokenizer, s, q, ["yes", "no"], use_prompt_perspective=True)
                return ["yes", "no"][r.answer_idx], r.fused.confidence
            seed_results["paec_prompt_simpletom"] = evaluate_simpletom(simpletom, paec_prompt_predict)

        # Phase 2: prefix-based PAEC
        if phase >= 2:
            print("\n--- Phase 2: Prefix-based PAEC ---")
            prefix_self_path = os.environ.get("PREFIX_SELF_PATH", "checkpoints/prefix_self")
            prefix_partner_path = os.environ.get("PREFIX_PARTNER_PATH", "checkpoints/prefix_partner")

            model_self, _ = load_model_with_prefix(config, prefix_self_path)
            model_partner, _ = load_model_with_prefix(config, prefix_partner_path)

            def paec_prefix_predict(s, q):
                r = runner.predict(model_self, model_partner, tokenizer, s, q, ["yes", "no"])
                return ["yes", "no"][r.answer_idx], r.fused.confidence
            seed_results["paec_prefix_simpletom"] = evaluate_simpletom(simpletom, paec_prefix_predict)

        results[seed] = seed_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"results_phase{phase}_{timestamp}.json")

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    config = PAECConfig.from_yaml(args.config)
    run_experiment(config, args.phase, args.output_dir)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_all_experiments.py
git commit -m "feat: experiment orchestrator script"
```

---

### Task 20: Final integration test

- [ ] **Step 1: Run all unit tests**

```bash
python -m pytest tests/ -v
```
Expected: All tests pass

- [ ] **Step 2: Test import chain**

```python
python -c "
from src.config import PAECConfig
from src.inference.logit_to_dirichlet import logits_to_opinion
from src.inference.ds_fusion import dempster_combine
from src.inference.perspective_runner import PerspectiveRunner
from src.evaluation.calibration import expected_calibration_error, brier_score
print('All imports successful')
"
```

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "chore: final integration — all modules importable and tested"
```

---

## Execution Summary

| Phase | Tasks | What it delivers |
|---|---|---|
| Chunk 1 (Tasks 1-5) | Foundation | Config, logit→Dirichlet, DS fusion, calibration — all tested |
| Chunk 2 (Tasks 6-11) | Data + Eval | ToMi/ExploreToM loaders, SimpleToM/CoordinationQA/ToMi evaluators |
| Chunk 3 (Tasks 12-14) | Models + Inference | Model loader, 3 baselines, perspective runner |
| Chunk 4 (Tasks 15-18) | Training + Notebooks | Prefix trainer, 3 Colab notebooks |
| Chunk 5 (Tasks 19-20) | Orchestration | run_all_experiments script, integration test |

**Total: 20 tasks, ~50 steps. Estimated: 2-3 sessions.**

Phase 1 experiments can run after Chunk 3 (no training needed).
Phase 2 experiments require Chunk 4 (prefix training on Colab).
