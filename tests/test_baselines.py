import pytest


def test_format_standard_prompt():
    from src.baselines.standard_prompting import format_prompt
    prompt = format_prompt("A story here.", "What does X believe?", ["Room A", "Room B"])
    assert "A story here" in prompt
    assert "Room A" in prompt
    assert "Room B" in prompt
    assert "A." in prompt
    assert "B." in prompt


def test_format_simtom_prompt():
    from src.baselines.simtom_prompting import format_simtom_prompt
    prompt = format_simtom_prompt("A story.", "What does Alice believe?", ["A", "B"], agent="Alice")
    assert "Alice" in prompt
    assert "observed" in prompt.lower() or "access" in prompt.lower()


def test_format_simtom_no_agent():
    from src.baselines.simtom_prompting import format_simtom_prompt
    prompt = format_simtom_prompt("A story.", "What?", ["A", "B"])
    assert "relevant agent" in prompt.lower()


def test_format_sc_prompts():
    from src.baselines.self_consistency import format_sc_prompts
    prompts = format_sc_prompts("A story.", "What?", ["A", "B"], n_samples=4)
    assert len(prompts) == 4
    assert "step by step" in prompts[0].lower()


def test_format_sc_prompts_different_n():
    from src.baselines.self_consistency import format_sc_prompts
    prompts = format_sc_prompts("Story", "Q?", ["X", "Y", "Z"], n_samples=8)
    assert len(prompts) == 8
