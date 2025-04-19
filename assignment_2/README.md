# iNaturalist Image Classification

This project implements two approaches for classifying images from the iNaturalist dataset:
1. **Part A**: Training a custom 5-layer CNN from scratch.
2. **Part B**: Fine-tuning a pre-trained model like ResNet50.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Code Structure](#code-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Part A: Custom CNN](#part-a-custom-cnn)
6. [Part B: Fine-Tuning ResNet50](#part-b-fine-tuning-resnet50)
7. [Report](#report)
8. [Conclusion](#conclusion)

## Project Overview

- **Part A**: Build and train a flexible 5-layer CNN from scratch with customizable:
  - Number/size of filters
  - Activation functions
  - Dense layer neurons
- **Part B**: Fine-tune a pre-trained ResNet50 model by:
  - Training last k layers and freezing previous layers

## Dataset

The iNaturalist dataset has 10 classes with:
- **10K training images** 
- **2K test images**

Preprocessing includes:
- Normalization (ImageNet stats)
- Data augmentation (flips, rotation, color jitter)
- Class-balanced train/val splits

## Code Structure

### Part A: Custom CNN
- **`config.py`**: Sweep configurations and Best CNN hyperparameters
- **`data_module.py`**: Data loading/augmentation (PyTorch Lightning)
- **`model.py`**: Flexible 5-layer CNN implementation
- **`sweep.py`**: code to run W&B hyperparameter sweeps
- **`train_and_plot.py`**: Training 5 layer CNN model with best hyperparameters and visualizing test results

### Part B: Fine-Tuning
- **`data_module.py`**: Data loading/augmentation (PyTorch Lightning)
- **`fine_tune.py`**: code for fine-tuning ResNet50 
- **`model.py`**: Contains `FineTunedResNet` class

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YashG2003/Deep-Learning-DA6401/tree/main/assignment_2
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ``` 

### Part A: Custom CNN

1. **Run Sweep**:
   ```bash
   python part_a/sweep.py
   ``` 
2. **Train best model and visualize test predictions**:
   ```bash
   python part_a/train_and_plot.py
   ``` 

### Part B: Fine-Tuning

1. **Fine Tune**:
   ```bash
   python part_b/fine_tune.py
   ``` 

## Report

I have made a detailed report using W&B with all the plots and my observations based on them. Please have a look. 

https://wandb.ai/yashgawande25-indian-institute-of-technology-madras/da6401_a2/reports/Report-DA6401-Assignment-2--VmlldzoxMjM2NTMxNg?accessToken=az5fa9r9qowgf53uwcjecchpob8pqosbrxqanfy0j3yfmrg6vs1ouqvmlqpyncf6

## Conclusion

This project demonstrates:

1. Custom CNNs can achieve good performance with proper hyperparameter tuning.

2. Fine-tuning large pre-trained models gives higher accuracy and is efficient for small datasets.

3. W&B effectively tracks experiments and hyperparameter sweeps.