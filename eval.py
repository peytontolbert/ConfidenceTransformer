import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from datasets import load_dataset
from main import ConfidenceEnhancedTransformer  # Import the class from main.py

# Load the trained model and tokenizer
model_name = 'confidence_model'
model_path = model_name
tokenizer_path = model_name
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

model = ConfidenceEnhancedTransformer.from_pretrained(model_path, attn_implementation="eager")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()

# Load the WikiText-2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Ensure the example text is not empty and preprocess it
example_text = dataset['text'][3].strip()  # Get the first example from the dataset and strip whitespace
if not example_text:
    raise ValueError("The example text from the dataset is empty. Please check the dataset.")

# Example usage
if __name__ == "__main__":
    prompt = "tefewafwef aoasdfsfasdfsadfdfasdsdijfoiwej"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Check if input_ids is empty
    if inputs.input_ids.size(1) == 0:
        raise ValueError("The input text resulted in an empty input_ids tensor. Please check the input text.")
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_dropout_samples=10  # Increased from 5 to 10
        )
    # Get the confidence score
    confidence_score = outputs['confidence_score'].item()
    ood_score = outputs['ood_score'].item()
    print(f"Refined Confidence Score: {confidence_score}")
    print(f"OOD Score: {ood_score}")

    # Decode and print the generated text (not part of the confidence mechanism)
    generated_text = tokenizer.decode(outputs['lm_logits'].argmax(-1).squeeze().tolist())
    print(f"OOD example: {prompt}")
    print(f"Generated Text: {generated_text}")

    # Evaluate on an in-distribution example from WikiText-2
    inputs = tokenizer(example_text, return_tensors='pt').to(device)
    
    # Check if input_ids is empty
    if inputs.input_ids.size(1) == 0:
        raise ValueError("The example text resulted in an empty input_ids tensor. Please check the example text.")
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_dropout_samples=10
        )
    # Get the confidence score for the in-distribution example
    confidence_score = outputs['confidence_score'].item()
    ood_score = outputs['ood_score'].item()
    print(f"In-Distribution Example - Refined Confidence Score: {confidence_score}")
    print(f"In-Distribution Example - OOD Score: {ood_score}")

    # Decode and print the generated text for the in-distribution example
    generated_text = tokenizer.decode(outputs['lm_logits'].argmax(-1).squeeze().tolist())
    print(f"example text: {example_text}")
    print(f"In-Distribution Example - Generated Text: {generated_text}")
