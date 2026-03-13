import torch
from typing import List, Tuple


def format_prompt(story: str, question: str, options: List[str]) -> str:
    """Format a standard MC prompt."""
    options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    return (
        f"Read the following story and answer the question.\n\n"
        f"Story: {story}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with just the letter (A, B, etc.):"
    )


def _extract_answer_logits(model, tokenizer, prompt: str, options: List[str], max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared helper: run forward pass and extract logits for answer tokens."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_logits = outputs.logits[0, -1, :]
    answer_letters = [chr(65 + i) for i in range(len(options))]
    answer_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in answer_letters]
    answer_logits = last_logits[answer_ids]

    return answer_logits, last_logits


def predict_standard(
    model,
    tokenizer,
    story: str,
    question: str,
    options: List[str],
    max_length: int = 512,
) -> Tuple[int, float]:
    """Single-pass prediction with argmax on answer token logits."""
    prompt = format_prompt(story, question, options)
    answer_logits, _ = _extract_answer_logits(model, tokenizer, prompt, options, max_length)

    probs = torch.softmax(answer_logits, dim=0)
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    return pred_idx, confidence
