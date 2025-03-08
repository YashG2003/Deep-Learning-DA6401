import numpy as np
from keras.datasets import fashion_mnist

# Load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

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

print("Dataset splits saved successfully!")
