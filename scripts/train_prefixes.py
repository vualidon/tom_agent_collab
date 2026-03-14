"""Train perspective prefixes on GPU server.

Usage:
    # Train both prefixes (default)
    python scripts/train_prefixes.py --config configs/default.yaml --output_dir checkpoints

    # Train only one perspective
    python scripts/train_prefixes.py --perspective self --output_dir checkpoints

    # Resume from a specific ToMi generation count
    python scripts/train_prefixes.py --tomi_stories 1000 --output_dir checkpoints
"""
import argparse
import gc
import os
import subprocess
import sys
import torch

from src.config import PAECConfig
from src.models.model_loader import load_base_model
from src.models.perspective_prefix import train_prefix
from src.data.prefix_training_data import combine_training_data


def generate_tomi_data(tomi_dir: str, num_stories: int = 1000):
    """Generate ToMi dataset if not already present."""
    data_dir = os.path.join(tomi_dir, "data")
    if os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, "train.txt")):
        print(f"ToMi data already exists at {data_dir}")
        return

    main_py = os.path.join(tomi_dir, "main.py")
    if not os.path.exists(main_py):
        print(f"ToMi main.py not found at {main_py}. Clone it first:")
        print(f"  git clone https://github.com/facebookresearch/ToMi.git {tomi_dir}")
        sys.exit(1)

    print(f"Generating ToMi data ({num_stories} stories)...")
    subprocess.run(
        [sys.executable, "main.py", "-n", str(num_stories), "-o", "data"],
        cwd=tomi_dir,
        check=True,
    )
    print("ToMi data generated.")


def main():
    parser = argparse.ArgumentParser(description="Train PAEC perspective prefixes")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory for prefix weights")
    parser.add_argument("--perspective", choices=["self", "partner", "both"], default="both",
                        help="Which perspective to train")
    parser.add_argument("--tomi_dir", default="external/ToMi", help="ToMi repo directory")
    parser.add_argument("--tomi_stories", type=int, default=1000, help="Number of ToMi stories to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split")
    args = parser.parse_args()

    config = PAECConfig.from_yaml(args.config)

    # Generate ToMi data if needed
    generate_tomi_data(args.tomi_dir, args.tomi_stories)

    # Load and prepare training data
    print("\nPreparing training data...")
    train_data, tomi_test = combine_training_data(args.tomi_dir, seed=args.seed)
    print(f"Training examples: {len(train_data)}")
    print(f"ToMi test examples: {len(tomi_test)}")

    if len(train_data) == 0:
        print("ERROR: No training data loaded. Check data loaders.")
        sys.exit(1)

    perspectives = ["self", "partner"] if args.perspective == "both" else [args.perspective]

    for perspective in perspectives:
        print(f"\n{'='*60}")
        print(f"Training {perspective} prefix")
        print(f"{'='*60}")

        # Load fresh base model for each prefix
        base_model, tokenizer = load_base_model(config)

        output_path = os.path.join(args.output_dir, f"prefix_{perspective}")
        model = train_prefix(
            config=config,
            train_data=train_data,
            perspective=perspective,
            output_dir=output_path,
            tokenizer=tokenizer,
            base_model=base_model,
        )

        # Cleanup GPU memory before next prefix
        del model, base_model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Saved {perspective} prefix to {output_path}")

    # Verify prefix diversity if both were trained
    if args.perspective == "both":
        print(f"\n{'='*60}")
        print("Verifying prefix diversity")
        print(f"{'='*60}")

        from src.models.model_loader import load_model_with_prefix

        model1, _ = load_model_with_prefix(config, os.path.join(args.output_dir, "prefix_self"))
        emb1 = model1.get_prompt(batch_size=1).detach().cpu().flatten()
        del model1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model2, _ = load_model_with_prefix(config, os.path.join(args.output_dir, "prefix_partner"))
        emb2 = model2.get_prompt(batch_size=1).detach().cpu().flatten()
        del model2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cosine_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        print(f"Prefix cosine similarity: {cosine_sim.item():.4f}")
        print("(Lower = more distinct perspectives)")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
