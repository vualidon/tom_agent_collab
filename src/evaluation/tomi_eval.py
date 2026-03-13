from typing import List, Dict, Callable, Tuple
import numpy as np


def evaluate_tomi(
    test_data: List[Dict],
    predict_fn: Callable[[str, str], Tuple[str, float]],
) -> Dict:
    """Evaluate on ToMi test split.

    Args:
        test_data: ToMi test split (perspective pairs with ground truth)
        predict_fn: (story, question) -> (predicted_answer, confidence)

    Returns:
        Dict with accuracy per story_type and overall
    """
    by_type = {}
    for item in test_data:
        stype = item.get("story_type", "unknown")
        if stype not in by_type:
            by_type[stype] = []
        by_type[stype].append(item)

    results = {}
    all_correct = 0
    all_total = 0
    all_confidences = []
    all_correctness = []

    for stype, items in by_type.items():
        correct = 0
        confidences = []
        correctness = []

        for item in items:
            story = item.get("story", "")
            question = item.get("question", "")
            expected = item.get("answer", item.get("belief_location", "")).strip().lower()

            pred, conf = predict_fn(story, question)
            pred = pred.strip().lower()

            is_correct = pred == expected or expected in pred
            correct += int(is_correct)
            confidences.append(conf)
            correctness.append(int(is_correct))

        acc = correct / len(items) if items else 0.0
        results[f"{stype}_accuracy"] = acc
        results[f"{stype}_count"] = len(items)
        results[f"{stype}_confidences"] = np.array(confidences)
        results[f"{stype}_correctness"] = np.array(correctness)

        all_correct += correct
        all_total += len(items)
        all_confidences.extend(confidences)
        all_correctness.extend(correctness)

    results["overall_accuracy"] = all_correct / all_total if all_total > 0 else 0.0
    results["all_confidences"] = np.array(all_confidences)
    results["all_correctness"] = np.array(all_correctness)

    return results
