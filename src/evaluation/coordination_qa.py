import json
import os
from typing import List, Dict, Callable, Tuple
import numpy as np


def load_coordination_qa(repo_dir: str) -> List[Dict]:
    """Load CoordinationQA questions from cloned llm_coordination repo.

    Searches for the CoordinationQA data files in known locations.
    """
    qa_data = []

    # Known paths in the llm_coordination repo
    candidate_paths = [
        os.path.join(repo_dir, "data", "coordination_qa.json"),
        os.path.join(repo_dir, "coordination_qa", "data.json"),
        os.path.join(repo_dir, "CoordinationQA"),
        os.path.join(repo_dir, "data"),
    ]

    for path in candidate_paths:
        if os.path.isfile(path) and path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                qa_data.extend(data)
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        qa_data.extend(v)
            if qa_data:
                return qa_data

    # Fallback: search recursively for JSON files with "qa" or "question" in name
    for root, dirs, files in os.walk(repo_dir):
        for fname in sorted(files):
            if fname.endswith(".json") and ("qa" in fname.lower() or "question" in fname.lower()):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        # Check if items look like QA data
                        sample = data[0]
                        if isinstance(sample, dict) and ("question" in sample or "prompt" in sample):
                            qa_data.extend(data)
                    elif isinstance(data, dict):
                        for v in data.values():
                            if isinstance(v, list) and len(v) > 0:
                                qa_data.extend(v)
                except (json.JSONDecodeError, KeyError):
                    continue

    return qa_data


def categorize_coordination_qa(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Split CoordinationQA into env_comprehension / tom / joint_planning."""
    categories = {"env_comprehension": [], "tom_reasoning": [], "joint_planning": []}
    uncategorized = []

    for item in data:
        dim = str(item.get("dimension", item.get("category", item.get("type", "")))).lower()

        if "env" in dim or "comprehension" in dim or "environment" in dim:
            categories["env_comprehension"].append(item)
        elif "tom" in dim or "theory" in dim or "mind" in dim or "belief" in dim:
            categories["tom_reasoning"].append(item)
        elif "plan" in dim or "joint" in dim or "coordination" in dim:
            categories["joint_planning"].append(item)
        else:
            uncategorized.append(item)

    # If nothing was categorized, put everything in a single group
    if not any(categories.values()) and uncategorized:
        categories["all"] = uncategorized

    return categories


def evaluate_coordination_qa(
    data: List[Dict],
    predict_fn: Callable[[str, List[str]], Tuple[int, float]],
) -> Dict:
    """Evaluate on CoordinationQA.

    Args:
        data: list of MC question dicts
        predict_fn: (question_text, options) -> (chosen_index, confidence)

    Returns:
        Dict with accuracy per dimension and overall
    """
    categories = categorize_coordination_qa(data)
    results = {}
    all_correct = 0
    all_total = 0
    all_confidences = []
    all_correctness = []

    for cat_name, cat_data in categories.items():
        if not cat_data:
            continue
        correct = 0
        confidences = []
        correctness = []

        for item in cat_data:
            question = item.get("question", item.get("prompt", ""))
            options = item.get("options", item.get("choices", []))
            correct_idx = item.get("answer_index", item.get("correct", item.get("answer", 0)))

            if isinstance(correct_idx, str):
                if correct_idx.isdigit():
                    correct_idx = int(correct_idx)
                else:
                    correct_idx = ord(correct_idx.upper()) - ord("A")

            pred_idx, conf = predict_fn(question, options)
            is_correct = pred_idx == correct_idx
            correct += int(is_correct)
            confidences.append(conf)
            correctness.append(int(is_correct))

        acc = correct / len(cat_data) if cat_data else 0.0
        results[f"{cat_name}_accuracy"] = acc
        results[f"{cat_name}_count"] = len(cat_data)
        results[f"{cat_name}_confidences"] = np.array(confidences)
        results[f"{cat_name}_correctness"] = np.array(correctness)

        all_correct += correct
        all_total += len(cat_data)
        all_confidences.extend(confidences)
        all_correctness.extend(correctness)

    results["overall_accuracy"] = all_correct / all_total if all_total > 0 else 0.0
    results["all_confidences"] = np.array(all_confidences)
    results["all_correctness"] = np.array(all_correctness)

    return results
