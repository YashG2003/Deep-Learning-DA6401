import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
from model import NeuralNetwork, load_data  

# ✅ Replace this with the correct sweep ID from W&B UI
SWEEP_ID = "dx34l1mw"

# ✅ Load dataset using model.py's function
train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data()

# ✅ Initialize W&B API
api = wandb.Api()

# ✅ Fetch runs from correct project
runs = api.runs("yashgawande25-indian-institute-of-technology-madras/fashion-mnist-nn", 
               filters={"sweep": SWEEP_ID})

if len(runs) == 0:
    raise ValueError("No runs found for this sweep. Check if the Sweep ID is correct.")

# ✅ Find the best run
best_run = max(runs, key=lambda run: run.summary.get("val_acc", 0))
best_config = best_run.config

print("Best Hyperparameters:", json.dumps(best_config, indent=4))

# ✅ Initialize W&B for new run
wandb.init(project="fashion-mnist-nn", 
          name="try",
          config=best_config)

# ✅ Create model instance using best config
model = NeuralNetwork(wandb.config)
batch_size = wandb.config.batch_size
num_batches = train_images.shape[0] // batch_size

# ✅ Training loop using class methods
for epoch in range(wandb.config.epochs):
    total_train_loss, total_train_acc = 0, 0

    for i in range(num_batches):
        X_batch = train_images[i*batch_size:(i+1)*batch_size]
        y_batch = train_labels[i*batch_size:(i+1)*batch_size]

        # Forward pass
        model.forward(X_batch)
        batch_loss = model.cross_entropy_loss(y_batch, model.activations[-1])
        batch_acc = np.mean(np.argmax(model.activations[-1], axis=1) == np.argmax(y_batch, axis=1))

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

# ✅ Test evaluation using class method
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_preds = np.argmax(model.forward(test_images), axis=1)
test_labels_true = np.argmax(test_labels, axis=1)

print(f"Test Accuracy: {test_acc:.4f}")

# Confusion matrix
cm = confusion_matrix(test_labels_true, test_preds)

# Visualization 
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

plt.savefig("try.png")
wandb.log({"try": wandb.Image("try.png")})
plt.show()

wandb.finish()





