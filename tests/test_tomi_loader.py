import pytest
from src.data.tomi_loader import parse_tomi_story, extract_perspective_pairs, parse_tomi_trace

SAMPLE_STORY = """Logan entered the den.
Avery entered the den.
The lettuce is in the green_crate.
Avery exited the den.
Logan moved the lettuce to the blue_box.
Where will Avery look for the lettuce?"""


def test_parse_tomi_story():
    result = parse_tomi_story(SAMPLE_STORY)
    assert "story" in result
    assert "question" in result
    assert "Avery" in result["question"]
    assert len(result["agents"]) >= 2
    assert "Logan" in result["agents"]
    assert "Avery" in result["agents"]


def test_parse_tomi_trace():
    trace = parse_tomi_trace("false_belief\t1\tAvery\tlettuce\tgreen_crate\tblue_box")
    assert trace["story_type"] == "false_belief"
    assert trace["agent"] == "Avery"
    assert trace["belief_location"] == "green_crate"
    assert trace["true_location"] == "blue_box"


def test_extract_perspective_pairs():
    story_data = {
        "story": SAMPLE_STORY.rsplit("\n", 1)[0],
        "question": "Where will Avery look for the lettuce?",
        "agents": ["Logan", "Avery"],
        "agent": "Avery",
        "belief_location": "green_crate",
        "true_location": "blue_box",
        "story_type": "false_belief",
    }
    pairs = extract_perspective_pairs(story_data)
    assert len(pairs) == 2
    assert pairs[0]["perspective"] == "self"
    assert pairs[1]["perspective"] == "partner"
    assert pairs[0]["answer"] == "green_crate"  # Avery's belief


def test_extract_preserves_original_question():
    story_data = {
        "story": "Some story",
        "question": "Where will Avery look for the lettuce?",
        "agents": ["Logan", "Avery"],
        "agent": "Avery",
        "belief_location": "green_crate",
        "true_location": "blue_box",
        "story_type": "false_belief",
    }
    pairs = extract_perspective_pairs(story_data)
    assert pairs[0]["question"] == "Where will Avery look for the lettuce?"


def test_extract_no_agents():
    story_data = {"story": "No agents here.", "question": "What?", "agents": []}
    pairs = extract_perspective_pairs(story_data)
    assert len(pairs) == 0
