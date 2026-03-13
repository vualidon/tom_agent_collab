import torch
from typing import List, Tuple
from src.baselines.standard_prompting import _extract_answer_logits


def format_simtom_prompt(
    story: str,
    question: str,
    options: List[str],
    agent: str = "",
) -> str:
    """Format a SimToM perspective-taking prompt."""
    options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

    if agent:
        perspective_instruction = (
            f"Consider only what {agent} has directly observed or been told. "
            f"Ignore any information that {agent} would not have access to. "
            f"Think step by step about what {agent} knows and doesn't know."
        )
    else:
        perspective_instruction = (
            "Consider only what the relevant agent has directly observed. "
            "Think step by step about what they know and don't know."
        )

    return (
        f"{perspective_instruction}\n\n"
        f"Story: {story}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with just the letter (A, B, etc.):"
    )


def predict_simtom(
    model,
    tokenizer,
    story: str,
    question: str,
    options: List[str],
    agent: str = "",
    max_length: int = 512,
) -> Tuple[int, float]:
    """SimToM prediction: single pass with perspective prompt."""
    prompt = format_simtom_prompt(story, question, options, agent=agent)
    answer_logits, _ = _extract_answer_logits(model, tokenizer, prompt, options, max_length)

    probs = torch.softmax(answer_logits, dim=0)
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    return pred_idx, confidence
