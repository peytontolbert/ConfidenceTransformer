# Confidence Enhanced Transformer

Confidence Enhanced Transformer extends GPT-2 with auxiliary heads for sequence-level confidence scoring and out-of-distribution (OOD) scoring.

The model keeps GPT-2's language modeling head and adds:

- a confidence head over pooled hidden states
- an OOD detector head over pooled hidden states
- normalized attention-entropy and Monte Carlo dropout uncertainty signals
- optional OOD labels during training

## Installation

```sh
git clone https://github.com/peytontolbert/ConfidenceTransformer.git
cd ConfidenceTransformer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with `source venv/bin/activate`.

## Quick Demo

```sh
python main.py
```

This loads GPT-2, runs one prompt, and prints:

- refined confidence score
- OOD score
- greedy token decode from language-model logits

The demo is guarded by `if __name__ == "__main__"`, so importing `ConfidenceEnhancedTransformer` does not download or run GPT-2.

## Training

```sh
python train.py --epochs 3 --batch-size 4 --block-size 128
```

The training script uses WikiText-2 by default. It computes:

- language modeling loss
- confidence loss from normalized model predictive entropy
- OOD loss only when explicit `ood_labels` are supplied
- expected calibration error from predictions compared with labels

Checkpoints and reliability diagrams are written to `confidence_model/` by default.

## Evaluation

```sh
python eval.py --model-path confidence_model
```

Use `--skip-wikitext` to avoid loading WikiText during evaluation.

## Testing

```sh
python -m unittest test.py
```

The tests use a tiny randomly initialized GPT-2 config, so they do not download pretrained assets.

## Model Output

`ConfidenceEnhancedTransformer.forward(...)` returns a dictionary with:

- `loss`
- `lm_loss`
- `confidence_loss`
- `ood_loss`
- `lm_logits`
- `logits`
- `confidence_score`
- `ood_score`
- `base_confidence_score`
- `variance_confidence`
- `avg_attention_entropy`

`logits` is an alias for `lm_logits` for compatibility with normal language-model workflows.
