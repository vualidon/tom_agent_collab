import pytest
import torch
from unittest.mock import MagicMock
from src.inference.perspective_runner import PerspectiveRunner
from src.config import PAECConfig


def _make_mock_model(vocab_size=151936, seq_len=10):
    """Create a mock model that returns random logits."""
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, seq_len, vocab_size)
    mock_model.return_value = mock_output
    mock_model.device = torch.device("cpu")
    return mock_model


def _make_mock_tokenizer():
    """Create a mock tokenizer."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    mock_tokenizer.encode = lambda x, add_special_tokens=False: [ord(x[0]) if x else 65]
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.eos_token = "<eos>"
    return mock_tokenizer


def test_runner_returns_fused_result():
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
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
    assert result.answer_text in ["A", "B"]


def test_runner_binary_options():
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    config = PAECConfig()
    runner = PerspectiveRunner(config)

    result = runner.predict(
        model_self=mock_model,
        model_partner=mock_model,
        tokenizer=mock_tokenizer,
        story="Test story",
        question="Is this true?",
        options=["yes", "no"],
    )
    assert result.answer_idx in [0, 1]


def test_runner_four_options():
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    config = PAECConfig()
    runner = PerspectiveRunner(config)

    result = runner.predict(
        model_self=mock_model,
        model_partner=mock_model,
        tokenizer=mock_tokenizer,
        story="Test",
        question="Choose:",
        options=["A", "B", "C", "D"],
    )
    assert result.answer_idx in [0, 1, 2, 3]
    assert result.answer_text in ["A", "B", "C", "D"]


def test_runner_prompt_perspective_mode():
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    config = PAECConfig()
    runner = PerspectiveRunner(config)

    result = runner.predict(
        model_self=mock_model,
        model_partner=mock_model,
        tokenizer=mock_tokenizer,
        story="Test",
        question="Q?",
        options=["A", "B"],
        use_prompt_perspective=True,
    )
    assert hasattr(result, "fused")
    assert result.opinion_self is not None
    assert result.opinion_partner is not None


def test_runner_has_uncertainty_signals():
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    config = PAECConfig()
    runner = PerspectiveRunner(config)

    result = runner.predict(
        model_self=mock_model,
        model_partner=mock_model,
        tokenizer=mock_tokenizer,
        story="Test",
        question="Q?",
        options=["A", "B"],
    )

    assert 0 <= result.fused.vacuity <= 1.0
    assert result.fused.normalized_conflict >= 0
    assert result.fused.confidence >= 0
    assert result.fused.projected_prob.shape == (2,)
