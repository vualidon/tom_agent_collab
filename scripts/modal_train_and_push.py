"""Cost-aware Modal training runner for PAEC (1 seed) with optional HF upload.

Usage examples:
    # 1) Preflight only (recommended first)
    modal run scripts/modal_train_and_push.py --mode preflight --gpu A10

    # 2) Full train (uses preflight recommendation) + push to Hugging Face
    modal run scripts/modal_train_and_push.py --mode train --gpu A10 \
        --hf_repo_id your-username/paec-qwen3-1.7b-prefixes
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import modal

REMOTE_PROJECT_ROOT = "/workspace/project"
if REMOTE_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_PROJECT_ROOT)

from src.modal_training_plan import (
    GPU_RATES_PER_SEC,
    TrialResult,
    choose_best_trial,
    estimate_hourly_cost_usd,
)


APP_NAME = "paec-train-prefixes"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_SECRET_NAME = "huggingface"

HF_CACHE_MOUNT = Path("/vol/cache")
DATA_MOUNT = Path("/vol/data")
ARTIFACTS_MOUNT = Path("/vol/artifacts")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "datasets>=2.19.0",
        "bitsandbytes>=0.43.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "accelerate>=0.28.0",
        "huggingface_hub>=0.23.0",
    )
    .add_local_dir(str(PROJECT_ROOT), remote_path=REMOTE_PROJECT_ROOT)
)

hf_cache_volume = modal.Volume.from_name("paec-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("paec-data", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("paec-artifacts", create_if_missing=True)

COMMON_VOLUMES = {
    HF_CACHE_MOUNT: hf_cache_volume,
    DATA_MOUNT: data_volume,
    ARTIFACTS_MOUNT: artifacts_volume,
}

COMMON_ENVS = {
    "HF_HOME": str(HF_CACHE_MOUNT / "huggingface"),
    "TRANSFORMERS_CACHE": str(HF_CACHE_MOUNT / "huggingface"),
    "HF_DATASETS_CACHE": str(HF_CACHE_MOUNT / "datasets"),
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONUNBUFFERED": "1",
}

GPU_OPTIONS = {"T4": "T4", "L4": "L4", "A10": "A10"}


@dataclass
class PreflightOutput:
    gpu: str
    gpu_total_vram_gib: float
    perspective: str
    candidates: List[TrialResult]
    best: Optional[TrialResult]
    estimated_hourly_cost_usd: float


def _prepare_runtime() -> None:
    os.chdir(REMOTE_PROJECT_ROOT)
    if REMOTE_PROJECT_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_ROOT)


def _ensure_tomi_data(tomi_dir: str, tomi_stories: int) -> None:
    tomi_path = Path(tomi_dir)
    tomi_path.parent.mkdir(parents=True, exist_ok=True)

    if not (tomi_path / "main.py").exists():
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/ToMi.git", str(tomi_path)],
            check=True,
        )

    data_dir = tomi_path / "data"
    existing_txt = list(data_dir.glob("*.txt")) if data_dir.exists() else []
    if existing_txt:
        print(f"Using existing ToMi data in {data_dir} ({len(existing_txt)} txt files).")
        return

    if not data_dir.exists() or not existing_txt:
        subprocess.run(
            [sys.executable, "main.py", "-n", str(tomi_stories), "-o", "data"],
            cwd=str(tomi_path),
            check=True,
        )


def _make_optimizer(model, config):
    if config.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)
        except ImportError:
            pass
    import torch

    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def _run_single_trial(
    *,
    config,
    train_data: List[dict],
    perspective: str,
    batch_size: int,
    grad_accum: int,
    preflight_steps: int,
    preflight_examples: int,
) -> TrialResult:
    import torch
    from torch.utils.data import DataLoader

    from src.models.model_loader import load_base_model
    from src.models.perspective_prefix import PerspectiveDataset, create_prefix_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Modal preflight.")

    filtered = [ex for ex in train_data if ex["perspective"] == perspective]
    filtered = filtered[: max(preflight_examples, batch_size * grad_accum)]

    base_model = None
    tokenizer = None
    model = None
    optimizer = None
    dataloader = None

    try:
        base_model, tokenizer = load_base_model(config)
        model = create_prefix_model(config, base_model)
        dataset = PerspectiveDataset(filtered, tokenizer, max_length=config.max_seq_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        if len(dataloader) == 0:
            return TrialResult(
                batch_size=batch_size,
                grad_accum=grad_accum,
                optimizer_steps=0,
                samples_per_sec=0.0,
                peak_vram_gib=0.0,
                oom=True,
            )

        optimizer = _make_optimizer(model, config)

        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        opt_steps = 0
        micro_steps = 0
        loader_iter = iter(dataloader)
        start = time.perf_counter()

        while opt_steps < preflight_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                batch = next(loader_iter)

            batch = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
            if (batch["labels"] != -100).sum().item() == 0:
                continue
            raw_loss = model(**batch).loss
            if not torch.isfinite(raw_loss):
                return TrialResult(
                    batch_size=batch_size,
                    grad_accum=grad_accum,
                    optimizer_steps=0,
                    samples_per_sec=0.0,
                    peak_vram_gib=0.0,
                    oom=True,
                )
            loss = raw_loss / grad_accum
            loss.backward()
            micro_steps += 1

            if micro_steps % grad_accum == 0:
                max_grad_norm = float(getattr(config, "max_grad_norm", 1.0))
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

        # Flush tail grads if present for faithful throughput accounting.
        if micro_steps % grad_accum != 0:
            max_grad_norm = float(getattr(config, "max_grad_norm", 1.0))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1

        elapsed = max(time.perf_counter() - start, 1e-6)
        samples = opt_steps * batch_size * grad_accum
        peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)

        return TrialResult(
            batch_size=batch_size,
            grad_accum=grad_accum,
            optimizer_steps=opt_steps,
            samples_per_sec=samples / elapsed,
            peak_vram_gib=peak_vram_gib,
            oom=False,
        )

    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            return TrialResult(
                batch_size=batch_size,
                grad_accum=grad_accum,
                optimizer_steps=0,
                samples_per_sec=0.0,
                peak_vram_gib=0.0,
                oom=True,
            )
        raise
    finally:
        del dataloader, optimizer, model, base_model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_preflight_impl(
    *,
    config_path: str,
    seed: int,
    tomi_stories: int,
    perspective: str,
    preflight_steps: int,
    preflight_examples: int,
    candidates: List[Tuple[int, int]],
    gpu_label: str,
    cpu_cores: float,
    memory_gib: float,
    region_multiplier: float,
) -> Dict:
    import torch

    _prepare_runtime()

    from src.config import PAECConfig
    from src.data.prefix_training_data import combine_training_data

    tomi_dir = str(DATA_MOUNT / "ToMi")
    _ensure_tomi_data(tomi_dir=tomi_dir, tomi_stories=tomi_stories)

    config = PAECConfig.from_yaml(config_path)
    train_data, _ = combine_training_data(tomi_dir=tomi_dir, seed=seed)

    props = torch.cuda.get_device_properties(0)
    total_vram_gib = props.total_memory / (1024**3)

    trials: List[TrialResult] = []
    for batch_size, grad_accum in candidates:
        trial = _run_single_trial(
            config=config,
            train_data=train_data,
            perspective=perspective,
            batch_size=batch_size,
            grad_accum=grad_accum,
            preflight_steps=preflight_steps,
            preflight_examples=preflight_examples,
        )
        trials.append(trial)

    # Keep a safety margin between short preflight peak and full-train peak.
    best = choose_best_trial(trials, total_vram_gib=total_vram_gib, max_vram_fraction=0.80)
    hourly_cost = estimate_hourly_cost_usd(
        gpu_rate_per_sec=GPU_RATES_PER_SEC[gpu_label],
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        region_multiplier=region_multiplier,
    )

    result = PreflightOutput(
        gpu=gpu_label,
        gpu_total_vram_gib=total_vram_gib,
        perspective=perspective,
        candidates=trials,
        best=best,
        estimated_hourly_cost_usd=hourly_cost,
    )
    return {
        "gpu": result.gpu,
        "gpu_total_vram_gib": result.gpu_total_vram_gib,
        "perspective": result.perspective,
        "estimated_hourly_cost_usd": result.estimated_hourly_cost_usd,
        "candidates": [asdict(t) for t in result.candidates],
        "best": asdict(result.best) if result.best else None,
    }


def _upload_to_hf(
    *,
    hf_repo_id: str,
    run_dir: Path,
    run_summary_path: Path,
    commit_suffix: str,
) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN not found in environment. Configure Modal secret first.")

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=hf_repo_id, repo_type="model")
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 404:
            api.create_repo(repo_id=hf_repo_id, repo_type="model", private=False, exist_ok=True)
        else:
            raise

    for prefix_name in ("prefix_self", "prefix_partner"):
        local_folder = run_dir / prefix_name
        if not local_folder.exists():
            raise FileNotFoundError(f"Missing prefix folder: {local_folder}")
        api.upload_folder(
            repo_id=hf_repo_id,
            repo_type="model",
            folder_path=str(local_folder),
            path_in_repo=prefix_name,
            ignore_patterns=["epoch_*", "**/epoch_*"],
            commit_message=f"Upload {prefix_name} ({commit_suffix})",
        )

    api.upload_file(
        repo_id=hf_repo_id,
        repo_type="model",
        path_or_fileobj=str(run_summary_path),
        path_in_repo=f"runs/{run_summary_path.name}",
        commit_message=f"Upload run summary ({commit_suffix})",
    )


def _load_adapter_vector(prefix_dir: Path):
    import torch

    safe_path = prefix_dir / "adapter_model.safetensors"
    bin_path = prefix_dir / "adapter_model.bin"

    state_dict = None
    if safe_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safe_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"Could not find adapter weights in {prefix_dir}")

    vectors = []
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            vectors.append(value.detach().float().flatten())

    if not vectors:
        raise RuntimeError(f"No floating-point adapter tensors found in {prefix_dir}")

    return torch.cat(vectors)


def _run_train_impl(
    *,
    config_path: str,
    seed: int,
    tomi_stories: int,
    run_name: str,
    preflight: Dict,
    num_workers: int,
    hf_repo_id: str,
) -> Dict:
    import torch

    _prepare_runtime()

    from src.config import PAECConfig
    from src.data.prefix_training_data import combine_training_data
    from src.models.model_loader import load_base_model
    from src.models.perspective_prefix import train_prefix

    best = preflight.get("best")
    if not best:
        raise RuntimeError("No feasible preflight candidate found. Adjust candidate list and rerun preflight.")

    batch_size = int(best["batch_size"])
    grad_accum = int(best["grad_accum"])

    config = PAECConfig.from_yaml(
        config_path,
        overrides={
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "num_workers": num_workers,
        },
    )

    tomi_dir = str(DATA_MOUNT / "ToMi")
    _ensure_tomi_data(tomi_dir=tomi_dir, tomi_stories=tomi_stories)
    train_data, tomi_test = combine_training_data(tomi_dir=tomi_dir, seed=seed)

    run_dir = ARTIFACTS_MOUNT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    self_dir = run_dir / "prefix_self"
    partner_dir = run_dir / "prefix_partner"

    started_at = datetime.utcnow().isoformat() + "Z"

    for perspective, out_dir in (("self", self_dir), ("partner", partner_dir)):
        done_marker = out_dir / "adapter_config.json"
        if done_marker.exists():
            continue
        base_model, tokenizer = load_base_model(config)
        model = train_prefix(
            config=config,
            train_data=train_data,
            perspective=perspective,
            output_dir=str(out_dir),
            tokenizer=tokenizer,
            base_model=base_model,
        )
        del model, base_model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        artifacts_volume.commit()

    # Diversity check (adapter-weight cosine similarity).
    emb1 = _load_adapter_vector(self_dir)
    emb2 = _load_adapter_vector(partner_dir)
    cosine_sim = float(
        torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    )

    finished_at = datetime.utcnow().isoformat() + "Z"
    summary = {
        "run_name": run_name,
        "seed": seed,
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": config_path,
        "selected_batch_size": batch_size,
        "selected_grad_accum": grad_accum,
        "num_workers": num_workers,
        "train_examples": len(train_data),
        "tomi_test_examples": len(tomi_test),
        "prefix_self_path": str(self_dir),
        "prefix_partner_path": str(partner_dir),
        "prefix_cosine_similarity": cosine_sim,
        "preflight": preflight,
    }

    summary_path = run_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    artifacts_volume.commit()

    if hf_repo_id:
        _upload_to_hf(
            hf_repo_id=hf_repo_id,
            run_dir=run_dir,
            run_summary_path=summary_path,
            commit_suffix=run_name,
        )

    return summary


def _parse_candidates(candidates_csv: str) -> List[Tuple[int, int]]:
    """Parse candidate pairs from 'b1:g1,b2:g2'."""
    out: List[Tuple[int, int]] = []
    for item in candidates_csv.split(","):
        item = item.strip()
        if not item:
            continue
        b_s, g_s = item.split(":")
        out.append((int(b_s), int(g_s)))
    if not out:
        raise ValueError("At least one candidate pair is required.")
    return out


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 3,
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
    volumes=COMMON_VOLUMES,
    env=COMMON_ENVS,
)
def preflight_t4(**kwargs):
    return _run_preflight_impl(gpu_label="T4", **kwargs)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 3,
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
    volumes=COMMON_VOLUMES,
    env=COMMON_ENVS,
)
def preflight_l4(**kwargs):
    return _run_preflight_impl(gpu_label="L4", **kwargs)


@app.function(
    image=image,
    gpu="A10",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 3,
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
    volumes=COMMON_VOLUMES,
    env=COMMON_ENVS,
)
def preflight_a10(**kwargs):
    return _run_preflight_impl(gpu_label="A10", **kwargs)


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 8,
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
    volumes=COMMON_VOLUMES,
    env=COMMON_ENVS,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def train_t4(**kwargs):
    return _run_train_impl(**kwargs)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 8,
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
    volumes=COMMON_VOLUMES,
    env=COMMON_ENVS,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def train_l4(**kwargs):
    return _run_train_impl(**kwargs)


@app.function(
    image=image,
    gpu="A10",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 8,
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
    volumes=COMMON_VOLUMES,
    env=COMMON_ENVS,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def train_a10(**kwargs):
    return _run_train_impl(**kwargs)


def _pick_remote_fn(mode: str, gpu: str):
    gpu = gpu.upper()
    if gpu not in GPU_OPTIONS:
        raise ValueError(f"Unsupported gpu={gpu}. Choose one of {sorted(GPU_OPTIONS)}")

    if mode == "preflight":
        return {"T4": preflight_t4, "L4": preflight_l4, "A10": preflight_a10}[gpu]
    if mode == "train":
        return {"T4": train_t4, "L4": train_l4, "A10": train_a10}[gpu]
    raise ValueError(f"Unsupported mode={mode}. Use preflight or train.")


@app.local_entrypoint()
def main(
    mode: str = "preflight",
    gpu: str = "A10",
    config_path: str = "configs/default.yaml",
    seed: int = 42,
    tomi_stories: int = 1000,
    perspective: str = "self",
    preflight_steps: int = 40,
    preflight_examples: int = 512,
    candidates: str = "4:8,8:4,16:2,32:1",
    num_workers: int = 4,
    hf_repo_id: str = "",
    run_name: str = "",
    region_multiplier: float = 1.0,
):
    """Run preflight benchmark or full training."""
    candidate_pairs = _parse_candidates(candidates)
    fn = _pick_remote_fn(mode, gpu)

    preflight_kwargs = {
        "config_path": config_path,
        "seed": seed,
        "tomi_stories": tomi_stories,
        "perspective": perspective,
        "preflight_steps": preflight_steps,
        "preflight_examples": preflight_examples,
        "candidates": candidate_pairs,
        "cpu_cores": 4.0,
        "memory_gib": 16.0,
        "region_multiplier": region_multiplier,
    }

    if mode == "preflight":
        result = fn.remote(**preflight_kwargs)
        print(json.dumps(result, indent=2))
        return

    # mode == train: rerun preflight first to enforce budget-aware config selection.
    preflight_fn = _pick_remote_fn("preflight", gpu)
    preflight_result = preflight_fn.remote(**preflight_kwargs)
    print("Preflight result:")
    print(json.dumps(preflight_result, indent=2))

    if not preflight_result.get("best"):
        raise RuntimeError("No feasible training candidate found in preflight.")

    if not run_name:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = f"seed{seed}_{gpu.lower()}_{ts}"

    train_kwargs = {
        "config_path": config_path,
        "seed": seed,
        "tomi_stories": tomi_stories,
        "run_name": run_name,
        "preflight": preflight_result,
        "num_workers": num_workers,
        "hf_repo_id": hf_repo_id,
    }
    result = fn.remote(**train_kwargs)
    print("Training result:")
    print(json.dumps(result, indent=2))
