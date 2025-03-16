import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from model import NeuralNetwork, load_splitted_data
from split_data import create_splits

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train neural network on Fashion MNIST')
    
    # Add arguments with best defaults from sweep results
    parser.add_argument('-wp', '--wandb_project', default='fashion-mnist-nn',
                       help='Project name for Weights & Biases')
    parser.add_argument('-we', '--wandb_entity', default='yashgawande25-indian-institute-of-technology-madras',
                       help='Wandb Entity name')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                       help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                       help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('-l', '--loss', default='cross_entropy', 
                       choices=['mean_squared_error', 'cross_entropy'],
                       help='Loss function to use')
    parser.add_argument('-o', '--optimizer', default='nadam', 
                       choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                       help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.95,
                       help='Momentum for optimizer')
    parser.add_argument('-beta', '--beta', type=float, default=0.95,
                       help='Beta for RMSprop')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.85,
                       help='Beta1 for Adam/Nadam')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.9,
                       help='Beta2 for Adam/Nadam')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-8,
                       help='Epsilon value')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('-w_i', '--weight_init', default='xavier',
                       choices=['random', 'xavier'], help='Weight initialization')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                       help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('-a', '--activation', default='relu',
                       choices=['identity', 'sigmoid', 'tanh', 'relu'],
                       help='Activation function')

    args = parser.parse_args()

    # Check and create data splits
    required_files = ["train_images.npy", "train_labels.npy",
                     "val_images.npy", "val_labels.npy",
                     "test_images.npy", "test_labels.npy"]
    
    if not all(os.path.exists(f) for f in required_files):
        print("Creating dataset splits...")
        create_splits(dataset_name='fashion_mnist')

    # Load data
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_splitted_data()

    # Initialize WandB
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
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
            X_batch = train_images[i*batch_size:(i+1)*batch_size]
            y_batch = train_labels[i*batch_size:(i+1)*batch_size]

            # Forward pass
            model.forward(X_batch)
            
            # Calculate loss
            if config.loss == 'cross_entropy':
                batch_loss = model.cross_entropy_loss(y_batch, model.activations[-1])
            else:
                batch_loss = model.mean_squared_error(y_batch, model.activations[-1])
                
            batch_acc = np.mean(np.argmax(y_batch, axis=1) == np.argmax(model.activations[-1], axis=1))

            # Backward pass
            grads_W, grads_B = model.backward(y_batch)
            model.update_parameters(grads_W, grads_B, epoch, X_batch, y_batch)

            total_train_loss += batch_loss
            total_train_acc += batch_acc

        # Validation
        val_loss, val_acc = model.evaluate(val_images, val_labels)

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_train_loss/num_batches,
            "train_acc": total_train_acc/num_batches,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # Test evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test_preds = np.argmax(model.forward(test_images), axis=1)
    test_labels_true = np.argmax(test_labels, axis=1)

    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels_true, test_preds)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(*np.meshgrid(range(10), range(10)), s=cm.ravel() * 10, c="blue", alpha=0.6)
    for i in range(10):
        for j in range(10):
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12, fontweight="bold")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - Test Set\nAccuracy: {test_acc:.4f}")
    
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

    wandb.finish()

if __name__ == "__main__":
    main()
