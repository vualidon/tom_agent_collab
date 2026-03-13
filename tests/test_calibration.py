import pytest
import numpy as np
from src.evaluation.calibration import (
    expected_calibration_error,
    brier_score,
    mcnemar_test,
    accuracy_when_confident,
)


def test_ece_perfect_calibration():
    probs = np.array([0.9, 0.9, 0.1, 0.1])
    correct = np.array([1, 1, 0, 0])
    ece = expected_calibration_error(probs, correct, n_bins=2)
    assert ece < 0.1


def test_ece_worst_calibration():
    probs = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([0, 0, 0, 0])
    ece = expected_calibration_error(probs, correct, n_bins=2)
    assert ece > 0.5


def test_brier_perfect():
    probs = np.array([1.0, 1.0, 0.0])
    correct = np.array([1, 1, 0])
    assert brier_score(probs, correct) < 1e-5


def test_brier_worst():
    probs = np.array([0.0, 0.0, 1.0])
    correct = np.array([1, 1, 0])
    assert brier_score(probs, correct) > 0.9


def test_mcnemar_identical():
    a = np.array([1, 0, 1, 0, 1])
    b = np.array([1, 0, 1, 0, 1])
    chi2, p = mcnemar_test(a, b)
    assert p > 0.5


def test_mcnemar_different():
    a = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    b = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    chi2, p = mcnemar_test(a, b)
    assert chi2 > 0


def test_accuracy_when_confident():
    probs = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 1, 0, 0])
    vacuity = np.array([0.1, 0.2, 0.7, 0.8])
    thresholds, accs, covs = accuracy_when_confident(probs, correct, vacuity)
    # At low threshold (only confident preds), accuracy should be high
    low_mask = thresholds <= 0.3
    if low_mask.any():
        assert accs[low_mask][-1] >= 0.5
