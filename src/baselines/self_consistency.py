import torch
from typing import List, Tuple
from collections import Counter
from src.baselines.standard_prompting import _extract_answer_logits


def format_sc_prompts(
    story: str,
    question: str,
    options: List[str],
    n_samples: int = 2,
) -> List[str]:
    """Generate n_samples prompts for self-consistency voting."""
    options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    base_prompt = (
        f"Read the following story and answer the question.\n\n"
        f"Story: {story}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Think step by step, then answer with just the letter (A, B, etc.):"
    )
    return [base_prompt] * n_samples


def predict_self_consistency(
    model,
    tokenizer,
    story: str,
    question: str,
    options: List[str],
    n_samples: int = 2,
    temperature: float = 0.7,
    max_length: int = 512,
) -> Tuple[int, float]:
    """Self-consistency: sample N times with temperature, majority vote."""
    prompts = format_sc_prompts(story, question, options, n_samples)
    votes = []

    answer_letters = [chr(65 + i) for i in range(len(options))]
    answer_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in answer_letters]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_logits = outputs.logits[0, -1, :]
        answer_logits = last_logits[answer_ids]

        if temperature > 0:
            probs = torch.softmax(answer_logits / temperature, dim=0)
            pred_idx = torch.multinomial(probs, 1).item()
        else:
            pred_idx = answer_logits.argmax().item()

        votes.append(pred_idx)

    vote_counts = Counter(votes)
    winner = vote_counts.most_common(1)[0]
    pred_idx = winner[0]
    confidence = winner[1] / n_samples

    return pred_idx, confidence
