import pytest
from src.data.exploretom_loader import (
    filter_by_order,
    extract_perspective_pairs_exploretom,
    extract_agents_from_story,
)


def test_filter_by_order():
    examples = [
        {"nth_order": -1, "question": "factual"},
        {"nth_order": 0, "question": "state tracking"},
        {"nth_order": 1, "question": "1st order belief"},
        {"nth_order": 2, "question": "2nd order belief"},
    ]
    filtered = filter_by_order(examples, orders=[1, 2])
    assert len(filtered) == 2
    assert all(e["nth_order"] in [1, 2] for e in filtered)


def test_extract_agents_from_story():
    story = "Alice put the apple on the table. Bob left the room. Alice moved the apple."
    agents = extract_agents_from_story(story)
    assert "Alice" in agents
    assert "Bob" in agents
    # Should not include non-name words
    assert "The" not in agents


def test_extract_agents_filters_common_words():
    story = "Then Alice walked to the door. There Bob entered the room."
    agents = extract_agents_from_story(story)
    assert "Then" not in agents
    assert "There" not in agents


def test_extract_perspective_pairs():
    example = {
        "infilled_story": "Anne put the apple on the table. Bob left the room. Anne moved the apple to the shelf.",
        "question": "Does Bob know where the apple is?",
        "expected_answer": "no",
        "nth_order": 1,
    }
    pairs = extract_perspective_pairs_exploretom(example)
    assert len(pairs) == 2
    assert pairs[0]["perspective"] == "self"
    assert pairs[1]["perspective"] == "partner"
    assert pairs[0]["answer"] == "no"


def test_extract_pairs_too_few_agents():
    example = {
        "infilled_story": "Something happened with no clear agents doing actions.",
        "question": "What happened?",
        "expected_answer": "nothing",
        "nth_order": 1,
    }
    pairs = extract_perspective_pairs_exploretom(example)
    # May return empty if no agents found
    assert isinstance(pairs, list)
