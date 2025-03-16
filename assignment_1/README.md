# Fashion MNIST Classification using Neural Networks

This project implements a neural network from scratch using NumPy to classify images from the Fashion MNIST dataset. The project includes data preprocessing, model training, hyperparameter tuning using Weights & Biases (W&B), and evaluation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Code Structure](#code-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Visualizing Dataset](#visualizing-dataset)
6. [Training the Model](#training-the-model)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Evaluating the Model](#evaluating-the-model)
9. [Report](#report) 
10. [Conclusion](#conclusion)

## Project Overview

The goal of this project is to build a neural network from scratch using NumPy to classify images from the Fashion MNIST dataset. The project includes:
- Data preprocessing and splitting into training, validation, and test sets.
- Implementation of a neural network with customizable layers, activation functions, and optimizers.
- Hyperparameter tuning using Weights & Biases (W&B).
- Evaluation of the model on the test set and visualization of results.

## Dataset

The Fashion MNIST dataset consists of 70,000 grayscale images of 10 fashion categories. Each image is 28x28 pixels. The dataset is split into:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

The dataset is preprocessed and split into training, validation, and test sets using the `split_data.py` script.

## Code Structure

The project consists of the following files:
- **`visualize_data.py`**: Script to visualize samples from the Fashion MNIST dataset.
- **`split_data.py`**: Script to preprocess, split and locally save the dataset as training, validation, and test sets.
- **`model.py`**: Contains the implementation of the neural network, including forward and backward propagation, weight initialization, and optimizers.
- **`main.py`**: Entry point for creating dataset splits and running the hyperparameter sweep.
- **`train.py`**: Script to train the neural network with customizable hyperparameters by passing arguements in command line. It also evaluates trained model on test data and plots the confusion matrix.
- **`sweep.py`**: Script to perform hyperparameter tuning using W&B sweep configuration.
- **`test.py`**: Script to evaluate the best model on the test set and visualize results using confusion matrix.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YashG2003/Deep-Learning-DA6401.git
   cd assignment_1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Weights & Biases**:
   * Create a W&B account at wandb.ai.
   * Log in to W&B using the command:
   ```bash
   wandb login
   ```

## Visualizing Dataset

To visualize samples from the Fashion MNIST dataset, run:
```bash
python visualize_data.py
```

This script will display a grid of sample images from the dataset and log the visualization to W&B.

## Training the Model

This script will:
1. Use the best default values for each hyperparameter, custom values can be passed as arguement.
2. Split the data, save it locally and train the model using these hyperparameters. Log the results to W&B.
3. Evaluate the model on the test set and log the results to W&B.
4. Generate and save a confusion matrix.

To train the model with default hyperparameters, run:
```bash
python train.py --wandb_entity myname --wandb_project myprojectname
```
wandb_entity: Wandb Entity (username) used to track experiments in the Weights & Biases dashboard.

wandb_project: Project name used to track experiments in Weights & Biases dashboard

You can customize the training process by passing command-line arguments. For example:
```bash
python train.py --epochs 20 --batch_size 64 --learning_rate 0.001 --optimizer nadam
```

## Hyperparameter Tuning

For splitting the data, saving it locally and performing hyperparameter tuning by W&B sweep using the same data every time, run:
```bash
python main.py
```

This script will launch a hyperparameter sweep using Bayesian optimization. The sweep configuration is defined in `sweep.py`.

## Evaluating the Model

To evaluate the best model on the test set, run:
```bash
python test.py
```

This script will:
1. Set the SWEEP_ID in the script. Load the best hyperparameters from the W&B sweep which you have run earlier.
2. Train the model using these hyperparameters.
3. Evaluate the model on the test set and log the results to W&B.
4. Generate and save a confusion matrix.

## Report

I have made a detailed report using W&B with all the plots and my observations based on them. Please have a look. 

https://api.wandb.ai/links/yashgawande25-indian-institute-of-technology-madras/yw6uqlre

## Conclusion

This project demonstrates the implementation of a neural network from scratch using NumPy for classifying images from the Fashion MNIST dataset. By leveraging Weights & Biases for hyperparameter tuning and visualization, we were able to efficiently explore various configurations and achieve competitive performance. The modular code structure allows for easy customization and extension to other datasets or neural network architectures.

Feel free to experiment with different hyperparameters, optimizers, and activation functions to further improve the model's performance. 