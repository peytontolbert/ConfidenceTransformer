import unittest

import torch
from transformers import GPT2Config

from main import ConfidenceEnhancedTransformer


class TestConfidenceEnhancedTransformer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = GPT2Config(
            vocab_size=64,
            n_positions=16,
            n_ctx=16,
            n_embd=32,
            n_layer=1,
            n_head=4,
            bos_token_id=0,
            eos_token_id=1,
            attn_pdrop=0.1,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
        )
        cls.model = ConfidenceEnhancedTransformer(config)
        cls.input_ids = torch.randint(0, config.vocab_size, (2, 8))
        cls.attention_mask = torch.ones_like(cls.input_ids)

    def test_initialization(self):
        self.assertIsInstance(self.model, ConfidenceEnhancedTransformer)

    def test_forward_pass(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                self.input_ids,
                attention_mask=self.attention_mask,
                num_dropout_samples=1,
            )

        self.assertIn("lm_logits", outputs)
        self.assertIn("logits", outputs)
        self.assertIn("confidence_score", outputs)
        self.assertIn("ood_score", outputs)
        self.assertEqual(outputs["lm_logits"].shape, (2, 8, 64))
        self.assertEqual(outputs["confidence_score"].shape, (2, 1))
        self.assertEqual(outputs["ood_score"].shape, (2,))

    def test_confidence_score_range(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                self.input_ids,
                attention_mask=self.attention_mask,
                num_dropout_samples=2,
            )

        self.assertTrue(torch.all(outputs["confidence_score"] >= 0.0))
        self.assertTrue(torch.all(outputs["confidence_score"] <= 1.0))

    def test_ood_score_range(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                self.input_ids,
                attention_mask=self.attention_mask,
                num_dropout_samples=1,
            )

        self.assertTrue(torch.all(outputs["ood_score"] >= 0.0))
        self.assertTrue(torch.all(outputs["ood_score"] <= 1.0))

    def test_loss_calculation_with_ood_labels(self):
        labels = self.input_ids.clone()
        ood_labels = torch.tensor([0.0, 1.0])
        outputs = self.model(
            self.input_ids,
            attention_mask=self.attention_mask,
            labels=labels,
            ood_labels=ood_labels,
            num_dropout_samples=1,
        )

        self.assertIsNotNone(outputs["loss"])
        self.assertIsNotNone(outputs["lm_loss"])
        self.assertIsNotNone(outputs["confidence_loss"])
        self.assertIsNotNone(outputs["ood_loss"])
        self.assertGreater(outputs["loss"].item(), 0.0)

    def test_training_step(self):
        model = ConfidenceEnhancedTransformer(self.model.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        labels = self.input_ids.clone()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            self.input_ids,
            attention_mask=self.attention_mask,
            labels=labels,
            num_dropout_samples=1,
        )
        outputs["loss"].backward()
        optimizer.step()

        self.assertGreater(outputs["loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
