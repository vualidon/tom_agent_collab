import numpy as np
from typing import Tuple, Callable


def expected_calibration_error(
    probs: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error."""
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


def accuracy_when_confident(
    probs: np.ndarray,
    correct: np.ndarray,
    vacuity: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute accuracy-coverage curve at different vacuity thresholds.

    Returns: (thresholds, accuracies, coverages)
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 20)

    accuracies = []
    coverages = []
    for tau in thresholds:
        mask = vacuity <= tau
        coverage = mask.sum() / len(vacuity)
        if mask.sum() > 0:
            acc = correct[mask].mean()
        else:
            acc = 0.0
        accuracies.append(float(acc))
        coverages.append(float(coverage))

    return thresholds, np.array(accuracies), np.array(coverages)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> Tuple[float, float]:
    """McNemar's test for paired comparisons between two methods.

    Args:
        correct_a: binary array, 1 if method A got it right
        correct_b: binary array, 1 if method B got it right

    Returns: (chi2_statistic, p_value)
    """
    from scipy.stats import chi2

    # b = A right, B wrong; c = A wrong, B right
    b = ((correct_a == 1) & (correct_b == 0)).sum()
    c = ((correct_a == 0) & (correct_b == 1)).sum()

    if b + c == 0:
        return 0.0, 1.0

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return float(chi2_stat), float(p_value)


def bootstrap_ci(
    metric_fn: Callable,
    probs: np.ndarray,
    correct: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.

    Returns: (mean, lower, upper)
    """
    rng = np.random.default_rng(seed)
    n = len(probs)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(probs[idx], correct[idx]))

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return (
        float(scores.mean()),
        float(np.percentile(scores, alpha * 100)),
        float(np.percentile(scores, (1 - alpha) * 100)),
    )
