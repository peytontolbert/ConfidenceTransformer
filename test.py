import unittest
import torch
from transformers import GPT2Tokenizer
from main import ConfidenceEnhancedTransformer

class TestConfidenceEnhancedTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.model = ConfidenceEnhancedTransformer.from_pretrained('gpt2')
        cls.model.eval()  # Set model to evaluation mode

    def test_initialization(self):
        self.assertIsInstance(self.model, ConfidenceEnhancedTransformer)

    def test_forward_pass(self):
        input_text = "What is the capital of USA?"
        input_tokens = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_tokens['input_ids'], num_dropout_samples=2)
        
        self.assertIn('lm_logits', outputs)
        self.assertIn('confidence_score', outputs)
        self.assertIn('ood_score', outputs)

    def test_confidence_score_range(self):
        input_text = "What is the capital of USA?"
        input_tokens = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_tokens['input_ids'], num_dropout_samples=2)
        
        confidence_score = outputs['confidence_score'].item()
        self.assertGreaterEqual(confidence_score, 0.0)
        self.assertLessEqual(confidence_score, 1.0)

    def test_ood_score_range(self):
        input_text = "What is the capital of USA?"
        input_tokens = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_tokens['input_ids'], num_dropout_samples=2)
        
        ood_score = outputs['ood_score'].item()
        self.assertGreaterEqual(ood_score, 0.0)
        self.assertLessEqual(ood_score, 1.0)

    def test_loss_calculation(self):
        input_text = "What is the capital of USA?"
        input_tokens = self.tokenizer(input_text, return_tensors="pt")
        labels = input_tokens['input_ids'].clone()
        with torch.no_grad():
            outputs = self.model(input_tokens['input_ids'], labels=labels, num_dropout_samples=2)
        
        self.assertIn('loss', outputs)
        self.assertIsNotNone(outputs['loss'])

    def test_training_step(self):
        input_text = "What is the capital of USA?"
        input_tokens = self.tokenizer(input_text, return_tensors="pt")
        labels = input_tokens['input_ids'].clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        self.model.train()
        optimizer.zero_grad()
        outputs = self.model(input_tokens['input_ids'], labels=labels, num_dropout_samples=2)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
