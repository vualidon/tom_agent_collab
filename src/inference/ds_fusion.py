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
    Falls back to averaging when conflict is near 1.0.
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
