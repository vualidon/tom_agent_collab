from dataclasses import dataclass
from typing import List, Optional


GPU_RATES_PER_SEC = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
}

CPU_RATE_PER_CORE_SEC = 0.0000131
MEMORY_RATE_PER_GIB_SEC = 0.00000222


@dataclass
class TrialResult:
    batch_size: int
    grad_accum: int
    optimizer_steps: int
    samples_per_sec: float
    peak_vram_gib: float
    oom: bool


def choose_best_trial(
    trials: List[TrialResult],
    total_vram_gib: float,
    max_vram_fraction: float = 0.95,
) -> Optional[TrialResult]:
    """Pick the fastest non-OOM trial under a VRAM ceiling.

    If none are under the VRAM ceiling, fallback to the fastest non-OOM trial.
    """
    non_oom = [t for t in trials if not t.oom]
    if not non_oom:
        return None

    vram_limit = total_vram_gib * max_vram_fraction
    under_limit = [t for t in non_oom if t.peak_vram_gib <= vram_limit]

    candidates = under_limit if under_limit else non_oom
    return max(candidates, key=lambda t: t.samples_per_sec)


def estimate_hourly_cost_usd(
    gpu_rate_per_sec: float,
    cpu_cores: float = 4.0,
    memory_gib: float = 8.0,
    region_multiplier: float = 1.0,
) -> float:
    """Estimate effective hourly cost including GPU + CPU + memory."""
    per_sec = (
        gpu_rate_per_sec
        + (cpu_cores * CPU_RATE_PER_CORE_SEC)
        + (memory_gib * MEMORY_RATE_PER_GIB_SEC)
    )
    return per_sec * 3600.0 * region_multiplier
