import pytest
from src.data.tomi_loader import parse_tomi_story, extract_perspective_pairs, parse_tomi_trace, split_tomi_stories

SAMPLE_STORY = """1 Logan entered the den.
2 Avery entered the den.
3 The lettuce is in the green_crate.
4 Avery exited the den.
5 Logan moved the lettuce to the blue_box.
6 Where will Avery look for the lettuce?\tgreen_crate\t1"""


def test_parse_tomi_story():
    result = parse_tomi_story(SAMPLE_STORY)
    assert "story" in result
    assert "question" in result
    assert "Avery" in result["question"]
    assert result["answer"] == "green_crate"
    assert len(result["agents"]) >= 2
    assert "Logan" in result["agents"]
    assert "Avery" in result["agents"]
    # Line numbers should be stripped
    assert not result["story"].startswith("1 ")


def test_parse_tomi_trace():
    trace = parse_tomi_trace("enter_agent_0,agent_1_enters,agent_1_exits,agent_0_moves_obj,first_order_1_tom,false_belief")
    assert trace["story_type"] == "false_belief"
    assert trace["question_type"] == "first_order_1_tom"


def test_split_tomi_stories():
    multi = """1 A entered the room.
2 B entered the room.
3 Where is A?\troom\t1
1 C entered the den.
2 D entered the den.
3 Where is C?\tden\t1"""
    blocks = split_tomi_stories(multi)
    assert len(blocks) == 2


def test_extract_perspective_pairs():
    story_data = {
        "story": "Logan entered the den.\nAvery entered the den.\nThe lettuce is in the green_crate.\nAvery exited the den.\nLogan moved the lettuce to the blue_box.",
        "question": "Where will Avery look for the lettuce?",
        "answer": "green_crate",
        "agents": ["Logan", "Avery"],
        "story_type": "false_belief",
    }
    pairs = extract_perspective_pairs(story_data)
    assert len(pairs) == 2
    assert pairs[0]["perspective"] == "self"
    assert pairs[1]["perspective"] == "partner"
    assert pairs[0]["answer"] == "green_crate"


def test_extract_preserves_original_question():
    story_data = {
        "story": "Some story",
        "question": "Where will Avery look for the lettuce?",
        "answer": "green_crate",
        "agents": ["Logan", "Avery"],
        "story_type": "false_belief",
    }
    pairs = extract_perspective_pairs(story_data)
    assert pairs[0]["question"] == "Where will Avery look for the lettuce?"


def test_extract_no_agents():
    story_data = {"story": "No agents here.", "question": "What?", "answer": "x", "agents": []}
    pairs = extract_perspective_pairs(story_data)
    assert len(pairs) == 0
