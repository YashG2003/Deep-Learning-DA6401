import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os
from model import NeuralNetwork, load_splitted_data
from split_data import create_splits

# Sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [10]},
        "weight_decay": {"values": [0]},
        "optimizer": {"values": ["nadam"]},
        "momentum": {"values": [0.95]},
        "beta": {"values": [0.95]},
        "beta1": {"values": [0.9]},
        "beta2": {"values": [0.999]},
        "epsilon": {"values": [1e-8]},
        "batch_size": {"values": [32]},
        "weight_init": {"values": ["xavier"]},
        "activation": {"values": ["relu"]},
        "num_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [64, 128, 256]},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
    },
}

def train():
    # Check and create data splits for MNIST
    required_files = ["train_images.npy", "train_labels.npy",
                      "val_images.npy", "val_labels.npy",
                      "test_images.npy", "test_labels.npy"]
    
    if not all(os.path.exists(f) for f in required_files):
        print("Creating dataset splits...")
        create_splits(dataset_name="mnist")  

    # Load data
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_splitted_data()

    # Initialize WandB
    wandb.init()
    config = wandb.config
    wandb.run.name = (f"hl_{config.num_layers}_{config.hidden_size}_"
                     f"bs_{config.batch_size}_ac_{config.activation}")

    # Create model
    model = NeuralNetwork(config)
    batch_size = config.batch_size
    num_batches = train_images.shape[0] // batch_size

    # Training loop
    for epoch in range(config.epochs):
        total_train_loss, total_train_acc = 0, 0

        for i in range(num_batches):
            X_batch = train_images[i * batch_size:(i + 1) * batch_size]
            y_batch = train_labels[i * batch_size:(i + 1) * batch_size]

            # Forward pass
            model.forward(X_batch)
            
            # Calculate loss and accuracy
            batch_loss = model.cross_entropy_loss(y_batch, model.activations[-1])
                
            batch_acc = np.mean(np.argmax(y_batch, axis=1) == np.argmax(model.activations[-1], axis=1))
            # Backward pass and parameter update
            grads_W, grads_B = model.backward(y_batch)
            model.update_parameters(grads_W, grads_B, epoch, X_batch, y_batch)

            total_train_loss += batch_loss
            total_train_acc += batch_acc

        # Validation evaluation
        val_loss, val_acc = model.evaluate(val_images, val_labels)

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_train_loss / num_batches,
            "train_acc": total_train_acc / num_batches,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # Test evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-nn")
    wandb.agent(sweep_id, function=train, count=10)
