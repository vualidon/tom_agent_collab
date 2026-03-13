from dataclasses import dataclass
from typing import List
import torch

from src.config import PAECConfig
from src.inference.logit_to_dirichlet import logits_to_opinion, Opinion
from src.inference.ds_fusion import dempster_combine, FusedResult


@dataclass
class PAECResult:
    answer_idx: int
    answer_text: str
    fused: FusedResult
    opinion_self: Opinion
    opinion_partner: Opinion
    raw_logits_self: torch.Tensor
    raw_logits_partner: torch.Tensor
    used_fallback: bool


class PerspectiveRunner:
    def __init__(self, config: PAECConfig):
        self.config = config

    def _get_answer_logits(
        self,
        model,
        tokenizer,
        prompt: str,
        options: List[str],
    ) -> torch.Tensor:
        """Run a single forward pass and extract logits for answer tokens."""
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_logits = outputs.logits[0, -1, :]

        answer_letters = [chr(65 + i) for i in range(len(options))]
        answer_ids = [
            tokenizer.encode(letter, add_special_tokens=False)[0]
            for letter in answer_letters
        ]
        return last_logits[answer_ids].float()  # ensure float32 for fusion math

    def _format_perspective_prompt(
        self,
        story: str,
        question: str,
        options: List[str],
        perspective: str,
    ) -> str:
        """Format prompt with perspective framing (used for prompt-only mode)."""
        options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

        if perspective == "self":
            prefix = (
                "Reason from the viewpoint of the agent being asked about. "
                "Consider only what they have directly seen or been told."
            )
        else:
            prefix = (
                "Reason from the viewpoint of the other agent. "
                "Consider only what they have directly seen or been told."
            )

        return (
            f"{prefix}\n\n"
            f"Story: {story}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Answer:"
        )

    def predict(
        self,
        model_self,
        model_partner,
        tokenizer,
        story: str,
        question: str,
        options: List[str],
        use_prompt_perspective: bool = False,
    ) -> PAECResult:
        """Run PAEC 2-pass inference and return fused result."""
        if use_prompt_perspective:
            prompt_self = self._format_perspective_prompt(story, question, options, "self")
            prompt_partner = self._format_perspective_prompt(story, question, options, "partner")
        else:
            options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            prompt = (
                f"Story: {story}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_str}\n\n"
                f"Answer:"
            )
            prompt_self = prompt
            prompt_partner = prompt

        # Pass 1: self-perspective
        logits_self = self._get_answer_logits(model_self, tokenizer, prompt_self, options)
        opinion_self = logits_to_opinion(
            logits_self,
            logit_transform=self.config.logit_transform,
            evidence_fn=self.config.evidence_fn,
        )

        # Pass 2: partner-perspective
        logits_partner = self._get_answer_logits(model_partner, tokenizer, prompt_partner, options)
        opinion_partner = logits_to_opinion(
            logits_partner,
            logit_transform=self.config.logit_transform,
            evidence_fn=self.config.evidence_fn,
        )

        # Fuse
        fused = dempster_combine(
            opinion_self,
            opinion_partner,
            conflict_epsilon=self.config.conflict_epsilon,
        )

        # Answer selection with degenerate fallback
        used_fallback = False
        if fused.vacuity > self.config.degenerate_vacuity_threshold:
            answer_idx = logits_self.argmax().item()
            used_fallback = True
        else:
            answer_idx = fused.projected_prob.argmax().item()

        return PAECResult(
            answer_idx=answer_idx,
            answer_text=chr(65 + answer_idx),
            fused=fused,
            opinion_self=opinion_self,
            opinion_partner=opinion_partner,
            raw_logits_self=logits_self,
            raw_logits_partner=logits_partner,
            used_fallback=used_fallback,
        )
