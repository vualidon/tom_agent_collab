import pytest
import torch
from src.inference.logit_to_dirichlet import logits_to_opinion


def test_uniform_logits_high_vacuity():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    opinion = logits_to_opinion(logits, logit_transform="none", evidence_fn="relu")
    assert opinion.belief.shape == (4,)
    assert abs(opinion.belief.sum().item() + opinion.vacuity - 1.0) < 1e-5
    assert torch.allclose(opinion.belief, opinion.belief[0].expand(4), atol=1e-5)


def test_confident_logits_low_vacuity():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    opinion = logits_to_opinion(logits, logit_transform="none", evidence_fn="relu")
    assert opinion.belief[0] > 0.5
    assert opinion.vacuity < 0.5


def test_all_negative_logits_center_min():
    logits = torch.tensor([-5.0, -3.0, -4.0, -6.0])
    opinion = logits_to_opinion(logits, logit_transform="center_min", evidence_fn="relu")
    assert opinion.belief[1] > opinion.belief[0]
    assert opinion.vacuity < 1.0


def test_all_negative_logits_no_centering_pure_vacuity():
    logits = torch.tensor([-5.0, -3.0, -4.0, -6.0])
    opinion = logits_to_opinion(logits, logit_transform="none", evidence_fn="relu")
    assert opinion.vacuity > 0.99


def test_softplus_no_hard_zeros():
    logits = torch.tensor([-5.0, -3.0, -4.0, -6.0])
    opinion = logits_to_opinion(logits, logit_transform="none", evidence_fn="softplus")
    assert opinion.vacuity < 1.0


def test_opinion_constraint():
    for _ in range(10):
        logits = torch.randn(4)
        opinion = logits_to_opinion(logits, logit_transform="center_min", evidence_fn="relu")
        assert abs(opinion.belief.sum().item() + opinion.vacuity - 1.0) < 1e-5


def test_binary_opinion():
    logits = torch.tensor([3.0, -1.0])
    opinion = logits_to_opinion(logits, logit_transform="center_min", evidence_fn="relu")
    assert opinion.belief.shape == (2,)
    assert opinion.belief[0] > opinion.belief[1]
