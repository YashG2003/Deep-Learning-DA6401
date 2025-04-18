import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb

# Initialize W&B run
wandb.init(project="fashion-mnist-nn", name="dataset_samples")

# Load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class labels in Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Create plot
fig = plt.figure(figsize=(10, 5))
for i in range(10):
    idx = np.where(train_labels == i)[0][0]  # Get index of first image of class i
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')

plt.tight_layout()

# Log to W&B
wandb.log({"Dataset Samples": wandb.Image(fig)})
plt.close(fig)  # Clean up memory

# Show plot locally
plt.show()
wandb.finish()
