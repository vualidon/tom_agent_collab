import os
import re
from typing import List, Dict, Tuple
import random


def parse_tomi_story(text: str) -> Dict:
    """Parse a ToMi story text into structured data."""
    lines = text.strip().split("\n")
    question = lines[-1]
    story = "\n".join(lines[:-1])

    agents = []
    for line in lines[:-1]:
        match = re.match(r"^(\w+) (entered|exited)", line)
        if match and match.group(1) not in agents:
            agents.append(match.group(1))

    return {"story": story, "question": question, "agents": agents}


def parse_tomi_trace(trace_line: str) -> Dict:
    """Parse a ToMi trace line for ground truth."""
    parts = trace_line.strip().split("\t")
    if len(parts) >= 6:
        return {
            "story_type": parts[0],
            "order": int(parts[1]),
            "agent": parts[2],
            "object": parts[3],
            "belief_location": parts[4],
            "true_location": parts[5],
        }
    return {}


def extract_perspective_pairs(story_data: Dict) -> List[Dict]:
    """Create self-perspective and partner-perspective training examples.

    Preserves the original ToMi question and uses trace data for ground truth.
    """
    agents = story_data.get("agents", [])
    if len(agents) < 2:
        return []

    original_question = story_data.get("question", "")
    queried_agent = story_data.get("agent", agents[0])
    belief_location = story_data.get("belief_location", "")
    true_location = story_data.get("true_location", "")
    story = story_data.get("story", "")

    pairs = []

    # Self perspective: the agent being asked about — answer is their belief
    pairs.append({
        "perspective": "self",
        "story": story,
        "question": original_question,
        "answer": belief_location if belief_location else true_location,
        "agent": queried_agent,
        "story_type": story_data.get("story_type", "unknown"),
    })

    # Partner perspective: the other agent — answer is what they would know
    other_agent = agents[1] if queried_agent == agents[0] else agents[0]
    pairs.append({
        "perspective": "partner",
        "story": story,
        "question": f"Where does {other_agent} think the object is?",
        "answer": true_location if true_location else belief_location,
        "agent": other_agent,
        "story_type": story_data.get("story_type", "unknown"),
    })

    return pairs


def load_tomi_dataset(
    tomi_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Load ToMi data and split into train/test perspective pairs."""
    data_dir = os.path.join(tomi_dir, "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"ToMi data directory not found: {data_dir}")

    all_pairs = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        txt_path = os.path.join(data_dir, fname)
        trace_path = txt_path.replace(".txt", ".trace")

        with open(txt_path) as f:
            stories_raw = f.read().strip().split("\n\n")
        trace_lines = []
        if os.path.exists(trace_path):
            with open(trace_path) as f:
                trace_lines = f.read().strip().split("\n")

        for i, story_text in enumerate(stories_raw):
            if not story_text.strip():
                continue
            parsed = parse_tomi_story(story_text)
            if i < len(trace_lines):
                trace = parse_tomi_trace(trace_lines[i])
                parsed.update(trace)
            pairs = extract_perspective_pairs(parsed)
            all_pairs.extend(pairs)

    random.seed(seed)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * (1 - test_ratio))
    return all_pairs[:split_idx], all_pairs[split_idx:]
