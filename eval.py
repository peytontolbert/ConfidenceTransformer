import argparse
from typing import Optional

import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

from main import ConfidenceEnhancedTransformer


def load_wikitext_example(index: int = 3) -> str:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    for text in dataset["text"][index:]:
        text = text.strip()
        if text:
            return text
    raise ValueError("Could not find a non-empty WikiText example.")


def score_text(
    model: ConfidenceEnhancedTransformer,
    tokenizer: GPT2Tokenizer,
    text: str,
    device: torch.device,
    num_dropout_samples: int,
) -> None:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    if inputs.input_ids.size(1) == 0:
        raise ValueError("Input text produced an empty input_ids tensor.")

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.get("attention_mask"),
            num_dropout_samples=num_dropout_samples,
        )

    generated_text = tokenizer.decode(outputs["lm_logits"].argmax(-1).squeeze().tolist())
    print(f"Input: {text}")
    print(f"Refined Confidence Score: {outputs['confidence_score'].item():.4f}")
    print(f"OOD Score: {outputs['ood_score'].item():.4f}")
    print(f"Greedy Token Decode: {generated_text}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ConfidenceEnhancedTransformer.")
    parser.add_argument("--model-path", default="confidence_model")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--prompt", default="tefewafwef aoasdfsfasdfsadfdfasdsdijfoiwej")
    parser.add_argument("--num-dropout-samples", type=int, default=10)
    parser.add_argument("--skip-wikitext", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path: Optional[str] = args.tokenizer_path or args.model_path
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = ConfidenceEnhancedTransformer.from_pretrained(
        args.model_path,
        attn_implementation="eager",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("OOD-style prompt")
    score_text(
        model=model,
        tokenizer=tokenizer,
        text=args.prompt,
        device=device,
        num_dropout_samples=args.num_dropout_samples,
    )

    if not args.skip_wikitext:
        print("WikiText example")
        score_text(
            model=model,
            tokenizer=tokenizer,
            text=load_wikitext_example(),
            device=device,
            num_dropout_samples=args.num_dropout_samples,
        )


if __name__ == "__main__":
    main()
