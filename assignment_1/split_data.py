import numpy as np
import os
from keras.datasets import fashion_mnist, mnist

def create_splits(dataset_name):
    # List of required files
    required_files = [
        "train_images.npy", "train_labels.npy",
        "val_images.npy", "val_labels.npy",
        "test_images.npy", "test_labels.npy"
    ]
    
    # Check if all files already exist
    if all(os.path.exists(f) for f in required_files):
        print("Dataset files already exist.")
        return False
    
    # --- Only execute below if files are missing ---
    # Load dataset
    if dataset_name=='fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    elif dataset_name=='mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape and normalize images
    train_images = train_images.reshape(-1, 784) / 255.0
    test_images = test_images.reshape(-1, 784) / 255.0

    # One-hot encode labels
    num_classes = 10
    train_labels = np.eye(num_classes)[train_labels]
    test_labels = np.eye(num_classes)[test_labels]

    # Set seed for reproducibility
    np.random.seed(42)

    # Shuffle indices once
    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)

    # Select 10% for validation
    val_size = int(0.1 * len(indices))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Split the dataset
    val_images, val_labels = train_images[val_indices], train_labels[val_indices]
    train_images, train_labels = train_images[train_indices], train_labels[train_indices]

    # Save splits
    np.save("train_images.npy", train_images)
    np.save("train_labels.npy", train_labels)
    np.save("val_images.npy", val_images)
    np.save("val_labels.npy", val_labels)
    np.save("test_images.npy", test_images)
    np.save("test_labels.npy", test_labels)
    
    return True

if __name__ == "__main__":
    if create_splits(dataset_name='fashion_mnist'):
        print("Dataset splits saved successfully!")
    else:
        print("Dataset files already exist. Skipping creation.")


