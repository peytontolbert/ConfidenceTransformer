import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

from main import ConfidenceEnhancedTransformer


class TextDataset(Dataset):
    def __init__(self, tokenizer: GPT2Tokenizer, texts: Iterable[str], block_size: int = 128):
        self.examples = []

        for text in texts:
            if not text.strip():
                continue
            tokenized_text = tokenizer.encode(text)
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(
                    torch.tensor(tokenized_text[i:i + block_size], dtype=torch.long)
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.examples[index]


def compute_sequence_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    predictions = shift_logits.argmax(dim=-1)
    valid = shift_labels.ne(-100)
    correct = predictions.eq(shift_labels) & valid
    accuracy = correct.sum(dim=-1).float() / valid.sum(dim=-1).clamp_min(1)
    return accuracy.detach().cpu().numpy()


def compute_ece(accuracies: np.ndarray, confidences: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_accuracy = np.mean(accuracies[in_bin])
            avg_confidence = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return float(ece)


def reliability_diagram(
    accuracies: np.ndarray,
    confidences: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    binned_accuracy = np.zeros(n_bins)
    binned_confidence = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(in_bin):
            binned_accuracy[i] = np.mean(accuracies[in_bin])
            binned_confidence[i] = np.mean(confidences[in_bin])

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    plt.plot(bin_centers, binned_accuracy, marker="o", label="Accuracy")
    plt.plot(bin_centers, binned_confidence, marker="s", label="Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def build_dataloaders(
    tokenizer: GPT2Tokenizer,
    block_size: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = dataset["text"]
    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)

    train_dataset = TextDataset(tokenizer, train_texts, block_size=block_size)
    val_dataset = TextDataset(tokenizer, val_texts, block_size=block_size)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset preparation produced no training or validation examples.")

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConfidenceEnhancedTransformer.")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--output-dir", default="confidence_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-dropout-samples", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = ConfidenceEnhancedTransformer.from_pretrained(
        args.model_name,
        attn_implementation="eager",
    )

    train_dataloader, val_dataloader = build_dataloaders(
        tokenizer,
        block_size=args.block_size,
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        print(f"Epoch {epoch + 1}/{args.epochs}")

        for batch in tqdm(train_dataloader, desc="Training"):
            inputs = batch.to(device)
            labels = inputs.clone()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=inputs,
                labels=labels,
                num_dropout_samples=args.num_dropout_samples,
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"Step {global_step}: loss={loss.item():.4f}, "
                    f"base_conf={outputs['base_confidence_score'].mean().item():.4f}, "
                    f"ood={outputs['ood_score'].mean().item():.4f}"
                )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = output_dir / f"step_{global_step}"
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

        avg_train_loss = epoch_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_accuracies = []
        all_confidences = []

        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, desc="Validation"):
                val_inputs = val_batch.to(device)
                val_labels = val_inputs.clone()
                val_outputs = model(
                    input_ids=val_inputs,
                    labels=val_labels,
                    num_dropout_samples=args.num_dropout_samples,
                )
                val_loss += val_outputs["loss"].item()

                batch_accuracies = compute_sequence_accuracy(
                    val_outputs["lm_logits"],
                    val_labels,
                )
                batch_confidences = val_outputs["confidence_score"].squeeze(-1).cpu().numpy()
                all_accuracies.extend(batch_accuracies.tolist())
                all_confidences.extend(batch_confidences.tolist())

        all_accuracies = np.array(all_accuracies)
        all_confidences = np.array(all_confidences)
        avg_val_loss = val_loss / len(val_dataloader)
        ece = compute_ece(all_accuracies, all_confidences, n_bins=10)

        print(f"Average Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")
        print(f"Expected Calibration Error after Epoch {epoch + 1}: {ece:.4f}")
        reliability_diagram(
            accuracies=all_accuracies,
            confidences=all_confidences,
            save_path=output_dir / f"reliability_epoch_{epoch + 1}.png",
            n_bins=10,
        )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
