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
    num_workers: int = 0

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
