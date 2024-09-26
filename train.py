import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.optim as optim
from datasets import load_dataset
from main import ConfidenceEnhancedTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 3
save_steps = 500  # Save the model every 500 steps

# Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, block_size=128):
        self.examples = []

        for text in texts:
            tokenized_text = tokenizer.encode(text)
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(
                    torch.tensor(tokenized_text[i:i + block_size], dtype=torch.long)
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = ConfidenceEnhancedTransformer.from_pretrained('gpt2')

# Load the WikiText-2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
texts = dataset['text']

# Split the dataset into training and validation sets
train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)

# Prepare datasets
train_dataset = TextDataset(
    tokenizer=tokenizer,
    texts=train_texts,
    block_size=128
)
val_dataset = TextDataset(
    tokenizer=tokenizer,
    texts=val_texts,
    block_size=128
)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Prepare optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define ECE computation function
def compute_ece(preds, confidences, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(preds[in_bin] == preds[in_bin])  # Adjust based on your prediction mechanism
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# Define Reliability Diagram function
def reliability_diagram(confidences, predictions, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    accuracy = np.zeros(n_bins)
    confidence = np.zeros(n_bins)
    prop = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop[i] = np.mean(in_bin)
        if prop[i] > 0:
            accuracy[i] = np.mean(predictions[in_bin] == predictions[in_bin])  # Replace with actual labels
            confidence[i] = np.mean(confidences[in_bin])

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, accuracy, marker='o', label='Accuracy')
    plt.plot(bin_centers, confidence, marker='s', label='Confidence')
    plt.fill_between(bin_boundaries[:-1], 0, 1, color='gray', alpha=0.1)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Reliability Diagram')
    plt.show()

# Training loop
model.train()
global_step = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0
    for batch in tqdm(train_dataloader):
        inputs = batch.to(device)
        labels = inputs.clone()

        optimizer.zero_grad()

        outputs = model(
            input_ids=inputs,
            labels=labels,
            num_dropout_samples=15  # You can adjust this number
        )
        loss = outputs['loss']
        loss.backward()

        # Check gradients for confidence_head and ood_detector
        for name, param in model.named_parameters():
            if 'confidence_head' in name or 'ood_detector' in name:
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad.mean().item()}")

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        global_step += 1

        # Log intermediate values for debugging
        base_confidence_score = outputs['base_confidence_score'].mean().item()
        variance_confidence = outputs['variance_confidence'].item()
        avg_attention_entropy = outputs['avg_attention_entropy'].mean().item()
        ood_score = outputs['ood_score'].mean().item()

        print(f"Step {global_step}: Loss = {loss.item():.4f}, "
              f"Base Confidence Score = {base_confidence_score:.4f}, "
              f"Variance Confidence = {variance_confidence:.4f}, "
              f"Avg Attention Entropy = {avg_attention_entropy:.4f}, "
              f"OOD Score = {ood_score:.4f}")

        # Save the model every 500 steps
        if global_step % save_steps == 0:
            model.save_pretrained(f'model_step_{global_step}.pth')
            tokenizer.save_pretrained(f'tokenizer_step_{global_step}.pth')

    avg_train_loss = epoch_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0
    all_val_preds = []
    all_val_confidences = []
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader, desc="Validation"):
            val_inputs = val_batch.to(device)
            val_labels = val_inputs.clone()

            val_outputs = model(
                input_ids=val_inputs,
                labels=val_labels,
                num_dropout_samples=15
            )
            val_loss += val_outputs['loss'].item()

            # Collect predictions and confidence scores for calibration
            val_confidences = val_outputs['base_confidence_score'].cpu().numpy()
            # Assuming you're using the model's logits to derive predictions
            val_logits = val_outputs['logits']
            val_preds = torch.argmax(val_logits, dim=-1).cpu().numpy()
            all_val_preds.extend(val_preds.flatten())
            all_val_confidences.extend(val_confidences.flatten())

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Average Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")

    # Calculate ECE
    ece = compute_ece(
        preds=np.array(all_val_preds),
        confidences=np.array(all_val_confidences),
        n_bins=10
    )
    print(f"Expected Calibration Error (ECE) after Epoch {epoch + 1}: {ece:.4f}")

    # Plot Reliability Diagram
    reliability_diagram(
        confidences=np.array(all_val_confidences),
        predictions=np.array(all_val_preds),
        n_bins=10
    )

    # Reset model to training mode
    model.train()

# Save the final trained model
model.save_pretrained('confidence_model')
tokenizer.save_pretrained('confidence_model')
