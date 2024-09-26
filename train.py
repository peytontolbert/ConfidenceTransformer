import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.optim as optim
from datasets import load_dataset
from main import ConfidenceEnhancedTransformer

num_epochs = 3
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

# Prepare dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    texts=texts,
    block_size=128
)

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

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

# Training loop
num_epochs = 3
model.train()
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
            num_dropout_samples=5  # You can adjust this number
        )
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Average loss: {avg_loss:.4f}")

# Save the trained model
model.save_pretrained('path_to_save_your_model')
tokenizer.save_pretrained('path_to_save_your_model')
