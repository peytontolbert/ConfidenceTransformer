import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.utils import ModelOutput


@dataclass
class ConfidenceModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    lm_logits: Optional[torch.Tensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    lm_loss: Optional[torch.Tensor] = None
    confidence_loss: Optional[torch.Tensor] = None
    ood_loss: Optional[torch.Tensor] = None
    confidence_score: Optional[torch.Tensor] = None
    ood_score: Optional[torch.Tensor] = None
    base_confidence_score: Optional[torch.Tensor] = None
    variance_confidence: Optional[torch.Tensor] = None
    avg_attention_entropy: Optional[torch.Tensor] = None


class ConfidenceEnhancedTransformer(GPT2LMHeadModel):
    """GPT-2 language model with confidence and OOD scoring heads."""

    def __init__(self, config):
        super().__init__(config)

        self.confidence_head = nn.Sequential(
            nn.Linear(config.n_embd, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.ood_detector = nn.Linear(config.n_embd, 1)

        self.confidence_head.apply(self._init_weights)
        self.ood_detector.apply(self._init_weights)

    @staticmethod
    def _masked_mean(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        mask = mask.unsqueeze(-1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _normalized_attention_entropy(
        attentions,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not attentions:
            return torch.zeros(batch_size, device=device)

        layer_entropies = []
        for attention in attentions:
            attention_probs = attention.mean(dim=1)
            entropy = -torch.sum(
                attention_probs * torch.log(attention_probs.clamp_min(1e-12)),
                dim=-1,
            )
            denom = math.log(max(attention_probs.size(-1), 2))
            layer_entropies.append((entropy / denom).mean(dim=-1))

        return torch.stack(layer_entropies, dim=0).mean(dim=0).clamp(0.0, 1.0)

    def _mc_dropout_confidence_variance(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        num_dropout_samples: int,
        reference_confidence: torch.Tensor,
    ) -> torch.Tensor:
        if num_dropout_samples <= 1:
            return torch.zeros_like(reference_confidence)

        original_mode = self.training
        dropout_scores = []

        try:
            self.train()
            with torch.no_grad():
                for _ in range(num_dropout_samples):
                    outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    pooled_hidden = self._masked_mean(
                        outputs.hidden_states[-1],
                        attention_mask,
                    )
                    dropout_scores.append(self.confidence_head(pooled_hidden))
        finally:
            self.train(original_mode)

        scores = torch.stack(dropout_scores, dim=0)
        return torch.var(scores, dim=0, unbiased=False)

    def _confidence_target_from_logits(
        self,
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(shift_logits.detach(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=-1)
        normalized_entropy = entropy / math.log(shift_logits.size(-1))
        token_confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0)

        valid = shift_labels.ne(-100)
        token_confidence = token_confidence * valid.to(token_confidence.dtype)
        return token_confidence.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ood_labels: Optional[torch.Tensor] = None,
        num_dropout_samples: int = 1,
        **kwargs,
    ) -> ConfidenceModelOutput:
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        lm_logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        pooled_hidden = self._masked_mean(hidden_states, attention_mask)

        base_confidence_score = self.confidence_head(pooled_hidden)
        variance_confidence = self._mc_dropout_confidence_variance(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_dropout_samples=num_dropout_samples,
            reference_confidence=base_confidence_score,
        )
        variance_penalty = (variance_confidence / 0.25).clamp(0.0, 1.0)

        avg_attention_entropy = self._normalized_attention_entropy(
            outputs.attentions,
            batch_size=input_ids.size(0),
            device=input_ids.device,
        )
        ood_score = torch.sigmoid(self.ood_detector(pooled_hidden)).squeeze(-1)

        refined_confidence_score = torch.stack(
            [
                base_confidence_score.squeeze(-1),
                1.0 - variance_penalty.squeeze(-1),
                1.0 - avg_attention_entropy,
                1.0 - ood_score,
            ],
            dim=-1,
        ).mean(dim=-1, keepdim=True)

        total_loss = None
        lm_loss = None
        confidence_loss = None
        ood_loss = None

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            confidence_target = self._confidence_target_from_logits(
                shift_logits,
                shift_labels,
            )
            confidence_loss = F.mse_loss(
                base_confidence_score.squeeze(-1),
                confidence_target,
            )

            if ood_labels is not None:
                ood_targets = ood_labels.to(device=ood_score.device, dtype=ood_score.dtype)
                ood_loss = F.binary_cross_entropy(ood_score, ood_targets)
            else:
                ood_loss = ood_score.sum() * 0.0

            total_loss = lm_loss + 0.5 * confidence_loss + 0.3 * ood_loss

        return ConfidenceModelOutput(
            loss=total_loss,
            logits=lm_logits,
            lm_logits=lm_logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            lm_loss=lm_loss,
            confidence_loss=confidence_loss,
            ood_loss=ood_loss,
            confidence_score=refined_confidence_score,
            ood_score=ood_score,
            base_confidence_score=base_confidence_score.squeeze(-1),
            variance_confidence=variance_confidence.squeeze(-1),
            avg_attention_entropy=avg_attention_entropy,
        )


def run_demo() -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = ConfidenceEnhancedTransformer.from_pretrained(
        "gpt2",
        attn_implementation="eager",
    )
    model.eval()

    input_text = "What is the capital of the USA?"
    input_tokens = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_tokens["input_ids"], num_dropout_samples=10)

    confidence_score = outputs["confidence_score"].item()
    ood_score = outputs["ood_score"].item()
    generated_text = tokenizer.decode(outputs["lm_logits"].argmax(-1).squeeze().tolist())

    print(f"Refined Confidence Score: {confidence_score:.4f}")
    print(f"OOD Score: {ood_score:.4f}")
    print(f"Generated Text: {generated_text}")


if __name__ == "__main__":
    run_demo()
