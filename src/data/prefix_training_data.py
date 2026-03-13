from typing import List, Dict, Tuple
import random
from collections import Counter

from src.data.tomi_loader import load_tomi_dataset
from src.data.exploretom_loader import load_and_prepare_exploretom


def validate_dataset(data: List[Dict], source_name: str) -> None:
    """Run data validation checks and print statistics."""
    print(f"\n=== {source_name} Dataset Validation ===")
    print(f"Total examples: {len(data)}")

    perspectives = Counter(d["perspective"] for d in data)
    print(f"Perspective distribution: {dict(perspectives)}")

    lengths = [len(d.get("story", "") + d.get("question", "")) for d in data]
    if lengths:
        lengths_sorted = sorted(lengths)
        n = len(lengths)
        print(f"Char lengths — mean: {sum(lengths)/n:.0f}, "
              f"p50: {lengths_sorted[n//2]}, "
              f"p95: {lengths_sorted[int(n*0.95)]}, "
              f"max: {lengths_sorted[-1]}")

    # Spot check 10 random examples
    if len(data) >= 10:
        samples = random.sample(data, 10)
        print("Spot check (10 random examples):")
        for s in samples:
            print(f"  [{s['perspective']}] Q: {s['question'][:80]}... A: {s.get('answer', 'N/A')[:40]}")


def combine_training_data(
    tomi_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Combine ToMi + ExploreToM into training and test sets.

    Returns:
        (train_data, test_data) — test_data is ToMi-only (for ToMi eval)
    """
    tomi_train, tomi_test = load_tomi_dataset(tomi_dir, test_ratio=test_ratio, seed=seed)
    validate_dataset(tomi_train, "ToMi (train)")
    validate_dataset(tomi_test, "ToMi (test)")

    exploretom = load_and_prepare_exploretom(orders=[1, 2])
    validate_dataset(exploretom, "ExploreToM")

    train_data = tomi_train + exploretom
    random.seed(seed)
    random.shuffle(train_data)

    print(f"\n=== Combined Training Set ===")
    print(f"Total: {len(train_data)} (ToMi: {len(tomi_train)}, ExploreToM: {len(exploretom)})")

    return train_data, tomi_test
