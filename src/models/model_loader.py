import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import PAECConfig


def resolve_device(device: str = "auto") -> str:
    """Resolve runtime device preference."""
    requested = (device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda but CUDA is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested device=mps but MPS is not available.")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    return requested


def _resolve_dtype(config: PAECConfig, resolved_device: str):
    if resolved_device in {"cuda", "mps"} and config.fp16:
        return torch.float16
    return torch.float32


def load_base_model(
    config: PAECConfig,
    device: str = "auto",
    use_gradient_checkpointing: bool | None = None,
):
    """Load base model with explicit device placement."""
    resolved_device = resolve_device(device)
    dtype = _resolve_dtype(config, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if resolved_device == "cuda":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    if resolved_device in {"mps", "cpu"}:
        model.to(resolved_device)

    if use_gradient_checkpointing is None:
        use_gradient_checkpointing = config.gradient_checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def load_model_with_prefix(
    config: PAECConfig,
    prefix_path: str,
    device: str = "auto",
    use_gradient_checkpointing: bool | None = False,
):
    """Load base model with a saved prefix checkpoint."""
    model, tokenizer = load_base_model(
        config,
        device=device,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    model = PeftModel.from_pretrained(model, prefix_path)
    model.eval()
    return model, tokenizer


def get_answer_token_ids(tokenizer, options: list) -> list:
    """Get token IDs for answer options (A, B, C, D or yes/no)."""
    token_ids = []
    for opt in options:
        ids = tokenizer.encode(opt, add_special_tokens=False)
        token_ids.append(ids[0] if ids else tokenizer.unk_token_id)
    return token_ids
