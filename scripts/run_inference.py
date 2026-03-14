"""Run PAEC inference demo on a few examples.

Usage:
    # Phase 1: prompt-only (no trained prefixes needed)
    python scripts/run_inference.py --config configs/default.yaml

    # Phase 2: with trained prefixes
    python scripts/run_inference.py --config configs/default.yaml \
        --prefix_self checkpoints/prefix_self \
        --prefix_partner checkpoints/prefix_partner
"""
import argparse
import gc
import torch

from src.config import PAECConfig
from src.models.model_loader import load_base_model, load_model_with_prefix
from src.inference.perspective_runner import PerspectiveRunner
from src.baselines.standard_prompting import predict_standard
from src.baselines.simtom_prompting import predict_simtom
from src.baselines.self_consistency import predict_self_consistency


DEMO_EXAMPLES = [
    {
        "story": "Alice put the ball in the basket. Alice left the room. Bob moved the ball to the box.",
        "question": "Where will Alice look for the ball?",
        "options": ["basket", "box"],
        "expected": "basket",
    },
    {
        "story": "Sally put the marble in the basket. Sally left. Anne moved the marble to the box. Sally returned.",
        "question": "Where will Sally look for the marble?",
        "options": ["basket", "box"],
        "expected": "basket",
    },
    {
        "story": "John put the chocolate in the cupboard. John left. Mary moved the chocolate to the drawer.",
        "question": "Where does John think the chocolate is?",
        "options": ["cupboard", "drawer"],
        "expected": "cupboard",
    },
]


def run_baselines(base_model, tokenizer, config, example):
    """Run all baselines on a single example."""
    story, question, options = example["story"], example["question"], example["options"]

    idx_std, conf_std = predict_standard(base_model, tokenizer, story, question, options)
    idx_sim, conf_sim = predict_simtom(base_model, tokenizer, story, question, options)
    idx_sc2, conf_sc2 = predict_self_consistency(
        base_model, tokenizer, story, question, options, n_samples=2, temperature=config.sc_temperature
    )

    print(f"  Standard:  {options[idx_std]} (conf={conf_std:.3f})")
    print(f"  SimToM:    {options[idx_sim]} (conf={conf_sim:.3f})")
    print(f"  SC-2:      {options[idx_sc2]} (conf={conf_sc2:.3f})")

    return {"standard": idx_std, "simtom": idx_sim, "sc2": idx_sc2}


def run_paec(runner, model_self, model_partner, tokenizer, example, use_prompt=False):
    """Run PAEC on a single example."""
    result = runner.predict(
        model_self=model_self,
        model_partner=model_partner,
        tokenizer=tokenizer,
        story=example["story"],
        question=example["question"],
        options=example["options"],
        use_prompt_perspective=use_prompt,
    )

    label = "PAEC(prompt)" if use_prompt else "PAEC(prefix)"
    options = example["options"]
    print(f"  {label}:  {options[result.answer_idx]} (conf={result.fused.confidence:.3f}, "
          f"vac={result.fused.vacuity:.3f}, conflict={result.fused.normalized_conflict:.3f})")
    print(f"    Self belief:    {[f'{b:.3f}' for b in result.opinion_self.belief]}")
    print(f"    Partner belief: {[f'{b:.3f}' for b in result.opinion_partner.belief]}")
    print(f"    Fused prob:     {[f'{p:.3f}' for p in result.fused.projected_prob]}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run PAEC inference demo")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prefix_self", default="", help="Path to self prefix checkpoint")
    parser.add_argument("--prefix_partner", default="", help="Path to partner prefix checkpoint")
    args = parser.parse_args()

    config = PAECConfig.from_yaml(args.config)
    runner = PerspectiveRunner(config)

    # Load base model
    print("Loading base model...")
    base_model, tokenizer = load_base_model(config)

    for i, example in enumerate(DEMO_EXAMPLES):
        print(f"\n{'='*60}")
        print(f"Example {i+1}: {example['question']}")
        print(f"Story: {example['story']}")
        print(f"Expected: {example['expected']}")
        print(f"{'='*60}")

        # Baselines
        run_baselines(base_model, tokenizer, config, example)

        # PAEC prompt-only
        run_paec(runner, base_model, base_model, tokenizer, example, use_prompt=True)

    # Phase 2: prefix-based PAEC
    if args.prefix_self and args.prefix_partner:
        print(f"\n\n{'#'*60}")
        print("Phase 2: Prefix-based PAEC")
        print(f"{'#'*60}")

        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Loading prefix models...")
        model_self, tokenizer = load_model_with_prefix(config, args.prefix_self)
        model_partner, _ = load_model_with_prefix(config, args.prefix_partner)

        for i, example in enumerate(DEMO_EXAMPLES):
            print(f"\n{'='*60}")
            print(f"Example {i+1}: {example['question']}")
            print(f"{'='*60}")
            run_paec(runner, model_self, model_partner, tokenizer, example, use_prompt=False)
    else:
        print("\n(Skipping Phase 2 — no prefix paths provided. Use --prefix_self and --prefix_partner)")

    print("\nDone!")


if __name__ == "__main__":
    main()
