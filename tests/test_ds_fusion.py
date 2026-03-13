import pytest
import torch
from src.inference.logit_to_dirichlet import Opinion
from src.inference.ds_fusion import dempster_combine


def _make_opinion(belief: list, vacuity: float) -> Opinion:
    b = torch.tensor(belief)
    K = len(belief)
    alpha = b * 10 + 1
    return Opinion(belief=b, vacuity=vacuity, alpha=alpha, prior=1.0 / K)


def test_agreeing_opinions():
    o1 = _make_opinion([0.7, 0.1], vacuity=0.2)
    o2 = _make_opinion([0.6, 0.1], vacuity=0.3)
    result = dempster_combine(o1, o2)
    assert result.projected_prob[0] > result.projected_prob[1]
    assert result.vacuity < min(o1.vacuity, o2.vacuity)


def test_conflicting_opinions():
    o1 = _make_opinion([0.8, 0.0], vacuity=0.2)
    o2 = _make_opinion([0.0, 0.8], vacuity=0.2)
    result = dempster_combine(o1, o2)
    assert result.normalized_conflict > 0.3


def test_total_conflict_fallback():
    o1 = _make_opinion([0.99, 0.0], vacuity=0.01)
    o2 = _make_opinion([0.0, 0.99], vacuity=0.01)
    result = dempster_combine(o1, o2, conflict_epsilon=1e-8)
    assert result.belief.shape == (2,)
    assert result.normalized_conflict >= 0.99


def test_one_vacuous_opinion():
    o1 = _make_opinion([0.8, 0.0], vacuity=0.2)
    o2 = _make_opinion([0.0, 0.0], vacuity=1.0)
    result = dempster_combine(o1, o2)
    assert result.projected_prob[0] > result.projected_prob[1]


def test_fused_constraint():
    o1 = _make_opinion([0.5, 0.2], vacuity=0.3)
    o2 = _make_opinion([0.3, 0.4], vacuity=0.3)
    result = dempster_combine(o1, o2)
    assert abs(result.belief.sum().item() + result.vacuity - 1.0) < 1e-4


def test_four_class_fusion():
    o1 = _make_opinion([0.5, 0.1, 0.1, 0.0], vacuity=0.3)
    o2 = _make_opinion([0.4, 0.2, 0.0, 0.1], vacuity=0.3)
    result = dempster_combine(o1, o2)
    assert result.belief.shape == (4,)
    assert result.projected_prob[0] > result.projected_prob[3]
