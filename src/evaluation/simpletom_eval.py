from typing import List, Dict, Callable, Tuple
import numpy as np


def load_simpletom() -> List[Dict]:
    """Load SimpleToM dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("allenai/SimpleToM", split="test")
    return [dict(row) for row in ds]


def categorize_questions(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Split SimpleToM into mental_state / behavior / judgment categories.

    Uses the dataset's column names to determine category. Falls back to
    question_order field if available.
    """
    categories = {"mental_state": [], "behavior": [], "judgment": []}

    for item in data:
        # Try explicit question_type field
        qtype = str(item.get("question_type", "")).lower()
        q_order = str(item.get("question_order", item.get("question_number", ""))).lower()

        if "mental" in qtype or "state" in qtype or q_order == "a" or q_order == "1":
            categories["mental_state"].append(item)
        elif "behavior" in qtype or "action" in qtype or q_order == "b" or q_order == "2":
            categories["behavior"].append(item)
        elif "judg" in qtype or "reason" in qtype or q_order == "c" or q_order == "3":
            categories["judgment"].append(item)
        else:
            # If no clear category, try to infer from question text
            q = item.get("question", "").lower()
            if "aware" in q or "know" in q or "realize" in q:
                categories["mental_state"].append(item)
            elif "will" in q or "would" in q or "going to" in q:
                categories["behavior"].append(item)
            elif "reasonable" in q or "rational" in q or "surprising" in q:
                categories["judgment"].append(item)
            else:
                categories["mental_state"].append(item)  # default

    return categories


def evaluate_simpletom(
    data: List[Dict],
    predict_fn: Callable[[str, str], Tuple[str, float]],
) -> Dict:
    """Evaluate a prediction function on SimpleToM.

    Args:
        data: SimpleToM dataset
        predict_fn: (story, question) -> (predicted_answer, confidence)

    Returns:
        Dict with accuracy per category and overall
    """
    categories = categorize_questions(data)
    results = {}

    all_confidences = []
    all_correctness = []

    for cat_name, cat_data in categories.items():
        if not cat_data:
            continue
        correct = 0
        confidences = []
        correctness = []

        for item in cat_data:
            story = item.get("story", item.get("narrative", ""))
            question = item.get("question", "")
            expected = item.get("answer", item.get("expected_answer", "")).strip().lower()

            pred, conf = predict_fn(story, question)
            pred = pred.strip().lower()

            is_correct = pred == expected or pred in expected or expected in pred
            correct += int(is_correct)
            confidences.append(conf)
            correctness.append(int(is_correct))

        acc = correct / len(cat_data) if cat_data else 0.0
        results[f"{cat_name}_accuracy"] = acc
        results[f"{cat_name}_count"] = len(cat_data)
        results[f"{cat_name}_confidences"] = np.array(confidences)
        results[f"{cat_name}_correctness"] = np.array(correctness)

        all_confidences.extend(confidences)
        all_correctness.extend(correctness)

    total_count = sum(results.get(f"{c}_count", 0) for c in categories)
    total_correct = sum(int(c) for c in all_correctness)
    results["overall_accuracy"] = total_correct / total_count if total_count > 0 else 0.0
    results["all_confidences"] = np.array(all_confidences)
    results["all_correctness"] = np.array(all_correctness)

    return results
