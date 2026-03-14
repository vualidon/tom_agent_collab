import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import PrefixTuningConfig, get_peft_model, TaskType
from tqdm import tqdm

from src.config import PAECConfig
from src.models.model_loader import load_base_model


class PerspectiveDataset(Dataset):
    """Dataset for prefix tuning with correct label masking.

    Uses right-padding during training (overrides model_loader's left-padding)
    to ensure answer token positions are consistent.
    """

    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncated_count = 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = (
            f"Story: {ex['story']}\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex.get('answer', '')}"
        )

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            padding_side="right",  # Right-pad for training (consistent token positions)
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Track truncation
        if len(self.tokenizer.encode(text, add_special_tokens=False)) > self.max_length:
            self.truncated_count += 1

        # Create labels: mask everything except answer tokens
        labels = input_ids.clone()

        # Find "Answer:" token position by encoding the prefix
        answer_prefix = (
            f"Story: {ex['story']}\n"
            f"Question: {ex['question']}\n"
            f"Answer: "
        )
        prefix_ids = self.tokenizer.encode(answer_prefix, add_special_tokens=False)
        answer_start = min(len(prefix_ids), self.max_length - 1)

        # Mask loss on prefix tokens (only train on answer)
        labels[:answer_start] = -100
        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_prefix_model(config: PAECConfig, base_model):
    """Wrap base model with prefix tuning."""
    # Prefix tuning is incompatible with gradient checkpointing (PEFT limitation)
    if getattr(base_model, "is_gradient_checkpointing", False):
        base_model.gradient_checkpointing_disable()
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=config.num_virtual_tokens,
        prefix_projection=config.prefix_projection,
        encoder_hidden_size=config.encoder_hidden_size,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model


def train_prefix(
    config: PAECConfig,
    train_data: list,
    perspective: str,
    output_dir: str,
    tokenizer=None,
    base_model=None,
):
    """Train a single perspective prefix.

    Args:
        config: PAEC config
        train_data: list of perspective-labeled examples
        perspective: "self" or "partner"
        output_dir: where to save the trained prefix
        tokenizer: pre-loaded tokenizer
        base_model: pre-loaded base model
    """
    if base_model is None or tokenizer is None:
        base_model, tokenizer = load_base_model(config)

    # Filter to matching perspective
    filtered = [ex for ex in train_data if ex["perspective"] == perspective]
    print(f"Training {perspective} prefix on {len(filtered)} examples")

    # Create prefix model
    model = create_prefix_model(config, base_model)

    # Dataset and DataLoader
    dataset = PerspectiveDataset(filtered, tokenizer, max_length=config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Optimizer
    if config.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)
        except ImportError:
            print("bitsandbytes not available, falling back to standard AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Scheduler
    total_steps = (len(dataloader) * config.num_epochs) // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    model.train()
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        # Report truncation statistics
        if hasattr(dataset, 'truncated_count') and dataset.truncated_count > 0:
            print(f"  Truncated examples: {dataset.truncated_count}")

        # Checkpoint
        if config.checkpoint_every_epoch:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            model.save_pretrained(epoch_dir)
            print(f"  Saved checkpoint to {epoch_dir}")

    # Save final
    model.save_pretrained(output_dir)
    print(f"Saved final {perspective} prefix to {output_dir}")

    return model
