from typing import List, Dict
import re


def load_exploretom_dataset() -> List[Dict]:
    """Load ExploreToM from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("facebook/ExploreToM", split="train")
    return [dict(row) for row in ds]


def filter_by_order(examples: List[Dict], orders: List[int] = None) -> List[Dict]:
    """Filter to specific ToM order questions.

    ExploreToM uses 'qprop=nth_order' as the column name.
    Values: -1 (memory), 0 (baseline), 1 (first-order), 2 (second-order).
    """
    if orders is None:
        orders = [1, 2]
    return [e for e in examples if e.get("qprop=nth_order") in orders]


def extract_agents_from_story(story: str) -> List[str]:
    """Extract agent names from an ExploreToM story.

    Uses a two-pass approach: first find capitalized words that appear as
    sentence subjects, then filter by frequency to distinguish names from
    location words.
    """
    agents = []
    # Match capitalized words that appear as subjects of action verbs
    action_verbs = (
        r"(?:put|moved|left|entered|took|went|said|told|picked|placed|"
        r"walked|ran|saw|heard|opened|closed|gave|received|grabbed|hid|"
        r"decided|thought|believed|knew|noticed|realized|returned|exited)"
    )
    for match in re.finditer(rf"\b([A-Z][a-z]{{2,}})\b(?=\s+{action_verbs})", story):
        name = match.group(1)
        # Filter out common non-name words
        if name.lower() not in {"the", "this", "that", "then", "there", "room", "after", "before", "when", "while"}:
            if name not in agents:
                agents.append(name)
    return agents


def extract_perspective_pairs_exploretom(example: Dict) -> List[Dict]:
    """Create perspective pairs from an ExploreToM example."""
    story = example.get("infilled_story", "")
    question = example.get("question", "")
    answer = str(example.get("expected_answer", ""))
    agents = extract_agents_from_story(story)

    if len(agents) < 2:
        # Fallback: extract agent names from the question
        q_agents = re.findall(r"\b([A-Z][a-z]{2,})\b", question)
        for a in q_agents:
            if a.lower() not in {"does", "will", "where", "what", "how", "room"} and a not in agents:
                agents.append(a)

    if len(agents) < 2:
        return []

    nth_order = example.get("qprop=nth_order", 1)

    pairs = [
        {
            "perspective": "self",
            "story": story,
            "question": f"From {agents[0]}'s perspective: {question}",
            "answer": answer,
            "agent": agents[0],
            "nth_order": nth_order,
        },
        {
            "perspective": "partner",
            "story": story,
            "question": f"From {agents[1]}'s perspective: {question}",
            "answer": answer,
            "agent": agents[1],
            "nth_order": nth_order,
        },
    ]
    return pairs


def load_and_prepare_exploretom(orders: List[int] = None) -> List[Dict]:
    """Load ExploreToM, filter by order, extract perspective pairs."""
    if orders is None:
        orders = [1, 2]
    raw = load_exploretom_dataset()
    filtered = filter_by_order(raw, orders=orders)
    all_pairs = []
    for example in filtered:
        pairs = extract_perspective_pairs_exploretom(example)
        all_pairs.extend(pairs)
    return all_pairs
