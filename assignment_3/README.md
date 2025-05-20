# ME21B062 Assignment 3
# English to Hindi Transliteration

This project implements sequence-to-sequence models for transliteration between Latin script (English) and Devanagari script (Hindi) using the Dakshina dataset. Two approaches are implemented:

- **Vanilla Seq2Seq**: A basic encoder-decoder architecture
- **Attention-based Seq2Seq**: Enhanced with attention mechanism for improved performance

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Setup and Installation](#setup-and-installation)
- [Vanilla Seq2Seq](#vanilla-seq2seq)
- [Attention-based Seq2Seq](#attention-based-seq2seq)
- [Visualizations](#visualizations)
- [Results](#results)

## Project Overview

### Vanilla Seq2Seq
Implements a basic encoder-decoder architecture with:
- Configurable embedding dimensions
- Flexible number of encoder/decoder layers
- Support for different RNN cell types (RNN, GRU, LSTM)
- Beam search decoding

### Attention-based Seq2Seq
Enhances the vanilla model with:
- Attention mechanism to focus on relevant input characters
- Improved handling of long sequences
- Better context for character-by-character generation

## Dataset
The Dakshina dataset contains pairs of words in:
- Latin script (romanized Hindi - English)
- Devanagari script (native Hindi)

The data is split into:
- Training set (45,000 word pairs)
- Development set (4,500 word pairs)
- Test set (4,500 word pairs)

The dataset is available here https://github.com/google-research-datasets/dakshina

## Code Structure
```
dakshina-transliteration/
├── data/                       # Data files, data loading and preprocessing
├── models/                     # Model architecture definitions
├── utils/                      # Training, evaluation, and visualization utilities
├── sweeps/                     # code for running wandb hyperparameter tuning sweeps
├── predictions_attention/      # predictions of attention model on test data
├── predictions_vanilla/        # predictions of vanilla model on test data
├── a3_notebook.ipynb           # jupyter notebook (kaggle) having entire code
├── main.py                     # Main entry point
├── train_vanilla.py            # Train vanilla seq2seq model
├── train_attention.py          # Train attention-based seq2seq model
├── predict.py                  # Generate predictions using trained models
└── README.md                   # Project documentation
```

### Key Components
- `data/dataset.py`: Handles loading and preprocessing of the Dakshina dataset
- `models/encoder.py`: Implements the encoder for both models
- `models/decoder.py`: Implements vanilla and attention-based decoders
- `models/attention.py`: Implements the attention mechanism
- `models/seq2seq.py`: Combines encoders and decoders into complete models
- `utils/train.py`: Training functions and utilities
- `utils/evaluate.py`: Evaluation metrics and prediction generation
- `utils/visualization.py`: Functions for visualizing attention heatmaps and neuron activations
- `sweeps/`: Contains code for running hyperparameter tuning sweeps using wandb
- `a3_notebook.ipynb`: jupyter notebook (kaggle) having entire code

## Setup and Installation

Clone the repository:
```bash
git clone https://github.com/YashG2003/Deep-Learning-DA6401.git
cd assignment_3
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the dataset if not present in data/ folder:
```bash
# Download from https://github.com/google-research-datasets/dakshina
# Place in the appropriate directory
```

## Vanilla Seq2Seq

Train the model with best hyperparameters:
```bash
python -m main --mode=train_vanilla
```

Run hyperparameter sweep:
```bash
python -m main --mode=sweep_vanilla
```

Generate predictions:
```bash
python -m main --mode=predict --model_path=best_vanilla_model.pt
```

## Attention-based Seq2Seq

Train the model with best hyperparameters:
```bash
python -m main --mode=train_attention
```

Run hyperparameter sweep:
```bash
python -m main --mode=sweep_attention
```

Generate predictions:
```bash
python -m main --mode=predict --model_path=best_attention_model.pt
```

## Visualizations

### Attention Heatmaps
Visualize which input characters the model focuses on when generating each output character:
```bash
python -m main --mode=visualize --visualize_type=attention --model_path=best_attention_model.pt
```

### Neuron Activations
Visualize how individual neurons in the model respond to different characters:
```bash
python -m main --mode=visualize --visualize_type=neuron --model_path=best_attention_model.pt --neuron_idx=78
```

## Results

The project demonstrates that:

- Attention mechanism significantly improves transliteration accuracy:
  - Vanilla Seq2Seq: ~38% word-level accuracy
  - Attention-based Seq2Seq: ~41% word-level accuracy

- Among the vanilla models, LSTM cells outperform both GRU and RNN for this task.

- Among the attention models, GRU cells perform better than both LSTM and RNN for this task.

- Beam search decoding with beam width = 3 improves results compared to greedy decoding

- Neuron visualization reveals specialized neurons that activate for specific character patterns