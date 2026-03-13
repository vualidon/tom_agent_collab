"""Orchestrate all PAEC experiments.

Usage:
    python scripts/run_all_experiments.py --config configs/default.yaml --phase 1
    python scripts/run_all_experiments.py --config configs/default.yaml --phase 2
"""
import argparse
import json
import os
import torch
import numpy as np
from datetime import datetime
from transformers import set_seed as hf_set_seed

from src.config import PAECConfig
from src.models.model_loader import load_base_model, load_model_with_prefix
from src.inference.perspective_runner import PerspectiveRunner
from src.baselines.standard_prompting import predict_standard
from src.baselines.simtom_prompting import predict_simtom
from src.baselines.self_consistency import predict_self_consistency
from src.evaluation.simpletom_eval import load_simpletom, evaluate_simpletom
from src.evaluation.tomi_eval import evaluate_tomi
from src.evaluation.coordination_qa import load_coordination_qa, evaluate_coordination_qa
from src.evaluation.calibration import expected_calibration_error, brier_score, mcnemar_test, bootstrap_ci
from src.data.prefix_training_data import combine_training_data


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_baselines_simpletom(base_model, tokenizer, simpletom_data, config):
    """Run all baselines on SimpleToM."""
    runner = PerspectiveRunner(config)
    results = {}

    # Baseline 1: Standard
    def std_fn(s, q):
        idx, c = predict_standard(base_model, tokenizer, s, q, ["yes", "no"])
        return ["yes", "no"][idx], c
    results["standard"] = evaluate_simpletom(simpletom_data, std_fn)

    # Baseline 2: SimToM
    def sim_fn(s, q):
        idx, c = predict_simtom(base_model, tokenizer, s, q, ["yes", "no"])
        return ["yes", "no"][idx], c
    results["simtom"] = evaluate_simpletom(simpletom_data, sim_fn)

    # Baseline 3: SC-2
    def sc2_fn(s, q):
        idx, c = predict_self_consistency(base_model, tokenizer, s, q, ["yes", "no"],
                                          n_samples=2, temperature=config.sc_temperature)
        return ["yes", "no"][idx], c
    results["sc2"] = evaluate_simpletom(simpletom_data, sc2_fn)

    # Baseline 4: SC-8
    def sc8_fn(s, q):
        idx, c = predict_self_consistency(base_model, tokenizer, s, q, ["yes", "no"],
                                          n_samples=8, temperature=config.sc_temperature)
        return ["yes", "no"][idx], c
    results["sc8"] = evaluate_simpletom(simpletom_data, sc8_fn)

    # Baseline 5: PAEC prompt-only
    def paec_p_fn(s, q):
        r = runner.predict(base_model, base_model, tokenizer, s, q, ["yes", "no"],
                          use_prompt_perspective=True)
        return ["yes", "no"][r.answer_idx], r.fused.confidence
    results["paec_prompt"] = evaluate_simpletom(simpletom_data, paec_p_fn)

    return results


def run_baselines_coordination_qa(base_model, tokenizer, cqa_data, config):
    """Run all baselines on CoordinationQA."""
    runner = PerspectiveRunner(config)
    results = {}

    def std_fn(q, opts):
        return predict_standard(base_model, tokenizer, "", q, opts)
    results["standard"] = evaluate_coordination_qa(cqa_data, std_fn)

    def sim_fn(q, opts):
        return predict_simtom(base_model, tokenizer, "", q, opts)
    results["simtom"] = evaluate_coordination_qa(cqa_data, sim_fn)

    def sc2_fn(q, opts):
        return predict_self_consistency(base_model, tokenizer, "", q, opts,
                                       n_samples=2, temperature=config.sc_temperature)
    results["sc2"] = evaluate_coordination_qa(cqa_data, sc2_fn)

    def sc8_fn(q, opts):
        return predict_self_consistency(base_model, tokenizer, "", q, opts,
                                       n_samples=8, temperature=config.sc_temperature)
    results["sc8"] = evaluate_coordination_qa(cqa_data, sc8_fn)

    def paec_p_fn(q, opts):
        r = runner.predict(base_model, base_model, tokenizer, "", q, opts,
                          use_prompt_perspective=True)
        return r.answer_idx, r.fused.confidence
    results["paec_prompt"] = evaluate_coordination_qa(cqa_data, paec_p_fn)

    return results


def run_experiment(config: PAECConfig, phase: int, output_dir: str,
                   prefix_self_path: str = "", prefix_partner_path: str = "",
                   tomi_dir: str = "external/ToMi", cqa_dir: str = "external/llm_coordination"):
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    simpletom_data = load_simpletom()
    print(f"SimpleToM: {len(simpletom_data)} examples")

    tomi_test = []
    try:
        _, tomi_test = combine_training_data(tomi_dir, seed=42)
        print(f"ToMi test: {len(tomi_test)} examples")
    except FileNotFoundError:
        print("ToMi data not found, skipping ToMi evaluation")

    cqa_data = load_coordination_qa(cqa_dir)
    print(f"CoordinationQA: {len(cqa_data)} examples")

    all_results = {}

    for seed in config.seeds:
        set_all_seeds(seed)
        print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")

        base_model, tokenizer = load_base_model(config)
        seed_results = {}

        # Phase 1: Baselines
        if phase >= 1:
            print("\n--- Phase 1: Baselines ---")
            seed_results["simpletom"] = run_baselines_simpletom(
                base_model, tokenizer, simpletom_data, config
            )
            for method, r in seed_results["simpletom"].items():
                print(f"  {method} SimpleToM: {r['overall_accuracy']:.3f}")

            if cqa_data:
                seed_results["coordination_qa"] = run_baselines_coordination_qa(
                    base_model, tokenizer, cqa_data, config
                )

        # Phase 2: Prefix PAEC
        if phase >= 2 and prefix_self_path and prefix_partner_path:
            print("\n--- Phase 2: Prefix PAEC ---")
            import gc
            del base_model; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            runner = PerspectiveRunner(config)
            model_self, tokenizer = load_model_with_prefix(config, prefix_self_path)
            model_partner, _ = load_model_with_prefix(config, prefix_partner_path)

            # SimpleToM
            def paec_pfx_fn(s, q):
                r = runner.predict(model_self, model_partner, tokenizer, s, q, ["yes", "no"])
                return ["yes", "no"][r.answer_idx], r.fused.confidence
            seed_results.setdefault("simpletom", {})["paec_prefix"] = evaluate_simpletom(simpletom_data, paec_pfx_fn)
            print(f"  PAEC(prefix) SimpleToM: {seed_results['simpletom']['paec_prefix']['overall_accuracy']:.3f}")

            # ToMi
            if tomi_test:
                seed_results["tomi_paec_prefix"] = evaluate_tomi(tomi_test, paec_pfx_fn)
                print(f"  PAEC(prefix) ToMi: {seed_results['tomi_paec_prefix']['overall_accuracy']:.3f}")

            del model_self, model_partner; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[seed] = seed_results

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"results_phase{phase}_{timestamp}.json")

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print(f"\n{'='*60}\nSUMMARY (mean across seeds)\n{'='*60}")
    for method in ["standard", "simtom", "sc2", "sc8", "paec_prompt", "paec_prefix"]:
        accs = []
        for seed in config.seeds:
            r = all_results.get(seed, {}).get("simpletom", {}).get(method, {})
            if "overall_accuracy" in r:
                accs.append(r["overall_accuracy"])
        if accs:
            print(f"  {method:<15} SimpleToM: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--prefix_self", default="")
    parser.add_argument("--prefix_partner", default="")
    parser.add_argument("--tomi_dir", default="external/ToMi")
    parser.add_argument("--cqa_dir", default="external/llm_coordination")
    args = parser.parse_args()

    config = PAECConfig.from_yaml(args.config)
    run_experiment(
        config, args.phase, args.output_dir,
        prefix_self_path=args.prefix_self,
        prefix_partner_path=args.prefix_partner,
        tomi_dir=args.tomi_dir,
        cqa_dir=args.cqa_dir,
    )
