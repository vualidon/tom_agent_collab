import os
import re
from typing import List, Dict, Tuple
import random


def split_tomi_stories(text: str) -> List[str]:
    """Split a ToMi file into individual story-question blocks.

    Each block starts with "1 " and ends with a question line containing tabs.
    Stories are NOT separated by blank lines — they're consecutive numbered blocks.
    """
    blocks = []
    current_lines = []

    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        # Detect start of new block: line starts with "1 "
        if re.match(r"^1\s", line) and current_lines:
            blocks.append("\n".join(current_lines))
            current_lines = []
        current_lines.append(line)

    if current_lines:
        blocks.append("\n".join(current_lines))

    return blocks


def parse_tomi_story(text: str) -> Dict:
    """Parse a ToMi story block into structured data.

    Format: numbered lines like "1 Oliver entered the porch."
    Last line: "9 Where will X look for Y?\tanswer\t1"
    """
    lines = text.strip().split("\n")

    # Last line has the question + answer (tab-separated)
    last_line = lines[-1]
    parts = last_line.split("\t")
    question_raw = parts[0]
    answer = parts[1].strip() if len(parts) >= 2 else ""

    # Strip line numbers from all lines: "1 Oliver entered..." -> "Oliver entered..."
    story_lines = []
    for line in lines[:-1]:
        stripped = re.sub(r"^\d+\s+", "", line)
        story_lines.append(stripped)
    question = re.sub(r"^\d+\s+", "", question_raw)

    story = "\n".join(story_lines)

    # Extract agent names from enter/exit actions
    agents = []
    for line in story_lines:
        match = re.match(r"^(\w+) (entered|exited)", line)
        if match and match.group(1) not in agents:
            agents.append(match.group(1))

    return {"story": story, "question": question, "answer": answer, "agents": agents}


def parse_tomi_trace(trace_line: str) -> Dict:
    """Parse a ToMi trace line for metadata.

    Trace format is comma-separated: action1,action2,...,question_type,story_type
    e.g.: "enter_agent_0,...,first_order_1_tom,false_belief"
    """
    parts = trace_line.strip().split(",")
    if len(parts) >= 2:
        story_type = parts[-1]  # e.g. "false_belief", "true_belief"
        question_type = parts[-2]  # e.g. "first_order_1_tom", "memory", "reality"
        return {
            "story_type": story_type,
            "question_type": question_type,
        }
    return {}


def extract_perspective_pairs(story_data: Dict) -> List[Dict]:
    """Create self-perspective and partner-perspective training examples."""
    agents = story_data.get("agents", [])
    if len(agents) < 2:
        return []

    question = story_data.get("question", "")
    answer = story_data.get("answer", "")
    story = story_data.get("story", "")
    story_type = story_data.get("story_type", "unknown")

    if not answer:
        return []

    # Determine which agent is being asked about from the question
    queried_agent = agents[0]
    for agent in agents:
        if agent in question:
            queried_agent = agent
            break

    other_agent = agents[1] if queried_agent == agents[0] else agents[0]

    pairs = []

    # Self perspective: the agent being asked about
    pairs.append({
        "perspective": "self",
        "story": story,
        "question": question,
        "answer": answer,
        "agent": queried_agent,
        "story_type": story_type,
    })

    # Partner perspective: the other agent's viewpoint
    pairs.append({
        "perspective": "partner",
        "story": story,
        "question": f"Where does {other_agent} think the object is?",
        "answer": answer,
        "agent": other_agent,
        "story_type": story_type,
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
            content = f.read()
        story_blocks = split_tomi_stories(content)

        trace_lines = []
        if os.path.exists(trace_path):
            with open(trace_path) as f:
                trace_lines = f.read().strip().split("\n")

        for i, story_text in enumerate(story_blocks):
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
