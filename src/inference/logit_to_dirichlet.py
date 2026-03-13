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
