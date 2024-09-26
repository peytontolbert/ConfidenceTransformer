import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from datasets import load_dataset

# Load the trained model and tokenizer
model_path = 'model_step_7500.pth'
tokenizer_path = 'tokenizer_step_7500.pth'
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

class ConfidenceEnhancedTransformer(GPT2LMHeadModel):
    def __init__(self, config):
        super(ConfidenceEnhancedTransformer, self).__init__(config)
        self.transformer = GPT2Model(config)
        #self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Language modeling head

        # Confidence scoring head for epistemic uncertainty and OOD detection
        self.confidence_head = nn.Sequential(
            nn.Linear(config.n_embd, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Single output for confidence score
            nn.Sigmoid()  # Confidence score between 0 and 1
        )
        
        # OOD detector head
        self.ood_detector = nn.Linear(config.n_embd, 1)  # Auxiliary head for OOD detection
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, num_dropout_samples=10, past_key_values=None):
        # Standard forward pass through transformer
        # Forward pass through GPT2LMHeadModel
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            past_key_values=past_key_values  # Pass the past_key_values argument
        )
        lm_logits = outputs.logits  # [batch_size, sequence_length, vocab_size]
        hidden_states = outputs.hidden_states[-1]  # Get the last hidden state
        attentions = outputs.attentions  # Attention weights

        # Language modeling logits
        #lm_logits = self.lm_head(hidden_states)

        # Base confidence score from hidden states
        base_confidence_score = self.confidence_head(hidden_states.mean(dim=1))

        # Attention-based confidence signal
        attention_entropy = []
        for attn_layer in attentions:
            attn_probs = attn_layer.mean(dim=1)  # Mean over heads
            attn_entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-12), dim=-1)
            attention_entropy.append(attn_entropy.mean(dim=-1))  # Mean over tokens
        avg_attention_entropy = torch.stack(attention_entropy).mean(dim=0)  # Mean over layers

        # Monte Carlo Dropout for variance estimation
        variance_confidence = 0.0
        dropout_scores = []
        if num_dropout_samples > 1:
            original_mode = self.training  # Save original mode
            self.train()  # Enable dropout layers
            for _ in range(num_dropout_samples):
                with torch.no_grad():
                    dropout_outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    dropout_hidden_states = dropout_outputs.hidden_states[-1]
                    dropout_confidence = self.confidence_head(dropout_hidden_states.mean(dim=1))
                    dropout_scores.append(dropout_confidence)

            self.train(original_mode)  # Restore original mode
            # Calculate variance of dropout predictions as a confidence measure
            dropout_scores = torch.stack(dropout_scores)
            variance_confidence = torch.var(dropout_scores, dim=0).mean()
        else:
            variance_confidence = torch.tensor(0.0)
        # OOD detection score
        ood_score = torch.sigmoid(self.ood_detector(hidden_states.mean(dim=1))).squeeze()

        # Adjust this line to handle ood_score shape correctly
        if len(ood_score.shape) == 0:
            ood_score = ood_score.unsqueeze(0)  # Add batch dimension if it's a scalar

        # Combine all signals into a refined confidence score
        refined_confidence_score = (
            base_confidence_score 
            - variance_confidence  # Lower confidence if high variance
            - avg_attention_entropy.unsqueeze(1)  # Lower confidence if high attention entropy
            - ood_score.unsqueeze(1)  # Lower confidence if high OOD score
        ).clamp(0, 1)  # Ensure the final score is between 0 and 1

        # Calculate total loss if labels are provided
        total_loss = None
        if labels is not None:
            # Language modeling loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Entropy-based confidence loss
            token_probs = F.softmax(shift_logits, dim=-1)
            entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-12), dim=-1).mean(dim=-1)
            confidence_loss = nn.MSELoss()(base_confidence_score.squeeze(), 1 - entropy)

            # OOD loss
            ood_loss = nn.BCELoss()(ood_score, torch.zeros_like(ood_score))  # Penalty for high OOD score

            # Adjust the weights for the confidence and OOD loss
            total_loss = lm_loss + 0.2 * confidence_loss + 0.05 * ood_loss

        return {
            'loss': total_loss,
            'lm_logits': lm_logits,
            'confidence_score': refined_confidence_score,  # Refined confidence score
            'ood_score': ood_score  # Out-of-distribution score
        }


model = ConfidenceEnhancedTransformer.from_pretrained(model_path, attn_implementation="eager")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()

# Load the WikiText-2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Ensure the example text is not empty and preprocess it
example_text = dataset['text'][1].strip()  # Get the first example from the dataset and strip whitespace
if not example_text:
    raise ValueError("The example text from the dataset is empty. Please check the dataset.")

# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time"
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
