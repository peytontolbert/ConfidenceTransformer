# Confidence Enhanced Transformer

This repository contains the implementation of a Confidence Enhanced Transformer model based on GPT-2. The model is designed to provide confidence scores for its predictions, detect out-of-distribution (OOD) inputs, and perform language modeling tasks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Testing](#testing)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Confidence Enhanced Transformer extends the GPT-2 model to include mechanisms for estimating the confidence of its predictions and detecting out-of-distribution inputs. This is achieved through additional neural network heads and Monte Carlo Dropout for uncertainty estimation.

## Features

- **Language Modeling**: The model can generate text based on input prompts.
- **Confidence Scoring**: Provides a confidence score for each prediction.
- **OOD Detection**: Detects out-of-distribution inputs.
- **Monte Carlo Dropout**: Uses dropout during inference to estimate prediction variance.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/confidence-enhanced-transformer.git
    cd confidence-enhanced-transformer
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run the `train.py` script. This script uses the WikiText-2 dataset for training.

