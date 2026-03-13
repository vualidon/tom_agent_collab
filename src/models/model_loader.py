import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import PAECConfig


def load_base_model(config: PAECConfig):
    """Load Qwen3-1.7B in fp16 with optional gradient checkpointing."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def load_model_with_prefix(config: PAECConfig, prefix_path: str):
    """Load base model with a saved prefix checkpoint."""
    model, tokenizer = load_base_model(config)
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
