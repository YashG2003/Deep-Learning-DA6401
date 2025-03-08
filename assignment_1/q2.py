'''
import numpy as np
import wandb
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Initialize wandb
wandb.init(project="fashion-mnist-nn", name="nn-training")

# Load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values to [0,1]
train_images = train_images.reshape(-1, 784) / 255.0  # Flatten to (num_samples, 784)
test_images = test_images.reshape(-1, 784) / 255.0

# One-hot encoding labels
num_classes = 10
train_labels = np.eye(num_classes)[train_labels]
test_labels = np.eye(num_classes)[test_labels]

# Neural Network Hyperparameters
input_size = 784
hidden_layers = [256, 64]  # Flexible number of hidden layers
output_size = 10
epochs = 20
learning_rate = 0.01
optimizer_type = "rmsprop"
batch_size = 32

# Initialize weights and biases
layers = [input_size] + hidden_layers + [output_size]
weights = [np.random.randn(layers[i], layers[i+1]) * 0.01 for i in range(len(layers)-1)]
biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward pass
def forward_pass(X):
    activations, hs = [X], []
    for W, b in zip(weights[:-1], biases[:-1]):
        h = np.dot(activations[-1], W) + b
        hs.append(h)
        activations.append(sigmoid(h))
    h_final = np.dot(activations[-1], weights[-1]) + biases[-1]
    hs.append(h_final)
    activations.append(softmax(h_final))
    return activations, hs

# Loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

# Backpropagation with different optimizers
history_w = [np.zeros_like(W) for W in weights]
history_b = [np.zeros_like(b) for b in biases]

squared_w = [np.zeros_like(W) for W in weights]
squared_b = [np.zeros_like(b) for b in biases]

momentum = 0.9
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8

# Backward pass
def backward_pass(X, y, activations, hs):
    global weights, biases, history_w, history_b, squared_w, squared_b
    m = X.shape[0]
    
    dH = activations[-1] - y
    grads_W = []
    grads_B = []
    
    for i in range(len(weights)-1, -1, -1):
        dW = np.dot(activations[i].T, dH) / m
        dB = np.sum(dH, axis=0, keepdims=True) / m
        grads_W.insert(0, dW)
        grads_B.insert(0, dB)
        
        if i != 0:
            dH = np.dot(dH, weights[i].T) * (activations[i] * (1 - activations[i]))
    
    # Update weights based on optimizer
    for i in range(len(weights)):
        if optimizer_type == "sgd":
            weights[i] -= learning_rate * grads_W[i]
            biases[i] -= learning_rate * grads_B[i]
        
        elif optimizer_type == "momentum":
            history_w[i] = momentum * history_w[i] + grads_W[i]
            history_b[i] = momentum * history_b[i] + grads_B[i]
            weights[i] -= learning_rate * history_w[i]
            biases[i] -= learning_rate * history_b[i]
            
        elif optimizer_type == "nesterov":
            # Lookahead position
            lookahead_w = weights[i] - momentum * history_w[i]
            lookahead_b = biases[i] - momentum * history_b[i]
            
            # Compute gradients at lookahead position
            grads_lookahead_W = np.copy(grads_W[i])  # Use provided grads directly
            grads_lookahead_B = np.copy(grads_B[i])

            # Update velocity
            history_w[i] = momentum * history_w[i] + learning_rate * grads_lookahead_W
            history_b[i] = momentum * history_b[i] + learning_rate * grads_lookahead_B
            
            # Update parameters
            weights[i] -= history_w[i]
            biases[i] -= history_b[i]

        
        elif optimizer_type == "rmsprop":
            squared_w[i] = beta2 * squared_w[i] + (1 - beta2) * grads_W[i]**2
            squared_b[i] = beta2 * squared_b[i] + (1 - beta2) * grads_B[i]**2
            weights[i] -= learning_rate * grads_W[i] / (np.sqrt(squared_w[i] + epsilon))
            biases[i] -= learning_rate * grads_B[i] / (np.sqrt(squared_b[i] + epsilon))
        
        elif optimizer_type in ["adam", "nadam"]:
            history_w[i] = beta1 * history_w[i] + (1 - beta1) * grads_W[i]
            history_b[i] = beta1 * history_b[i] + (1 - beta1) * grads_B[i]
            squared_w[i] = beta2 * squared_w[i] + (1 - beta2) * grads_W[i]**2
            squared_b[i] = beta2 * squared_b[i] + (1 - beta2) * grads_B[i]**2
            
            v_hat_w = history_w[i] / (1 - beta1**(epoch+1))
            v_hat_b = history_b[i] / (1 - beta1**(epoch+1))
            s_hat_w = squared_w[i] / (1 - beta2**(epoch+1))
            s_hat_b = squared_b[i] / (1 - beta2**(epoch+1))
            
            weights[i] -= learning_rate * v_hat_w / (np.sqrt(s_hat_w) + epsilon)
            biases[i] -= learning_rate * v_hat_b / (np.sqrt(s_hat_b) + epsilon)

# Training loop
for epoch in range(epochs):
    activations, hs = forward_pass(train_images)
    train_loss = cross_entropy_loss(train_labels, activations[-1])
    backward_pass(train_images, train_labels, activations, hs)
    
    test_activations, _ = forward_pass(test_images)
    test_loss = cross_entropy_loss(test_labels, test_activations[-1])
    
    # Compute accuracy
    train_preds = np.argmax(activations[-1], axis=1)
    train_true = np.argmax(train_labels, axis=1)
    train_acc = np.mean(train_preds == train_true)
    
    test_preds = np.argmax(test_activations[-1], axis=1)
    test_true = np.argmax(test_labels, axis=1)
    test_acc = np.mean(test_preds == test_true)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "test_acc": test_acc})


wandb.finish()
'''


import numpy as np
import wandb
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

# Initialize wandb sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [20]},
        "hidden_sizes": {"values": [[512, 128, 32], [512, 256, 128, 32], [512, 256, 128, 64, 32]]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "beta1": {"values": [0.9, 0.85]},
        "beta2": {"values": [0.999, 0.98]},
        "momentum": {"values": [0.9, 0.95]},
        "batch_size": {"values": [32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["relu", "sigmoid", "tanh"]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-nn")

# Load pre-saved dataset splits
train_images = np.load("train_images.npy")
train_labels = np.load("train_labels.npy")
val_images = np.load("val_images.npy")
val_labels = np.load("val_labels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")

# Now proceed with model training as usual
print("Loaded pre-split dataset!")


# Activation functions
def activation_function(x, func):
    if func == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif func == "tanh":
        return np.tanh(x)
    elif func == "relu":
        return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

def forward_pass(X, weights, biases, activation):
    activations, hs = [X], []
    for W, b in zip(weights[:-1], biases[:-1]):
        h = np.dot(activations[-1], W) + b
        hs.append(h)
        activations.append(activation_function(h, activation))
    h_final = np.dot(activations[-1], weights[-1]) + biases[-1]
    activations.append(softmax(h_final))
    return activations, hs

def backward_pass(activations, hs, weights, y_true, activation):
    dH = activations[-1] - y_true
    grads_W, grads_B = [], []
    for i in range(len(weights) - 1, -1, -1):
        dW = np.dot(activations[i].T, dH) / y_true.shape[0]
        dB = np.sum(dH, axis=0, keepdims=True) / y_true.shape[0]
        grads_W.insert(0, dW)
        grads_B.insert(0, dB)
        if i != 0:
            dH = np.dot(dH, weights[i].T) * (activations[i] * (1 - activations[i]))
    return grads_W, grads_B

def update_weights(weights, biases, grads_W, grads_B, config, opt_states, epoch):
    learning_rate, optimizer = config.learning_rate, config.optimizer
    history_w, history_b, squared_w, squared_b = opt_states
    beta1, beta2, epsilon, momentum = config.beta1, config.beta2, 1e-8, config.momentum

    for i in range(len(weights)):
        if optimizer == "sgd":
            weights[i] -= learning_rate * grads_W[i]
            biases[i] -= learning_rate * grads_B[i]
        elif optimizer == "momentum":
            history_w[i] = momentum * history_w[i] + grads_W[i]
            history_b[i] = momentum * history_b[i] + grads_B[i]
            weights[i] -= learning_rate * history_w[i]
            biases[i] -= learning_rate * history_b[i]
        elif optimizer == "nesterov":
            # Lookahead position
            lookahead_w = weights[i] - momentum * history_w[i]
            lookahead_b = biases[i] - momentum * history_b[i]

            # Compute gradients at lookahead position
            activations, hs = forward_pass(train_images, lookahead_w, lookahead_b, config.activation)
            grads_lookahead_W, grads_lookahead_B = backward_pass(activations, hs, weights, train_labels, config.activation)

            # Update velocity
            history_w[i] = momentum * history_w[i] + learning_rate * grads_lookahead_W[i]
            history_b[i] = momentum * history_b[i] + learning_rate * grads_lookahead_B[i]

            # Update parameters
            weights[i] -= history_w[i]
            biases[i] -= history_b[i]
        elif optimizer == "rmsprop":
            squared_w[i] = beta2 * squared_w[i] + (1 - beta2) * grads_W[i]**2
            squared_b[i] = beta2 * squared_b[i] + (1 - beta2) * grads_B[i]**2
            weights[i] -= learning_rate * grads_W[i] / (np.sqrt(squared_w[i] + epsilon))
            biases[i] -= learning_rate * grads_B[i] / (np.sqrt(squared_b[i] + epsilon))
        elif optimizer == "adam":
            history_w[i] = beta1 * history_w[i] + (1 - beta1) * grads_W[i]
            history_b[i] = beta1 * history_b[i] + (1 - beta1) * grads_B[i]
            squared_w[i] = beta2 * squared_w[i] + (1 - beta2) * grads_W[i]**2
            squared_b[i] = beta2 * squared_b[i] + (1 - beta2) * grads_B[i]**2
            v_hat_w = history_w[i] / (1 - beta1**(epoch+1))
            v_hat_b = history_b[i] / (1 - beta1**(epoch+1))
            s_hat_w = squared_w[i] / (1 - beta2**(epoch+1))
            s_hat_b = squared_b[i] / (1 - beta2**(epoch+1))
            weights[i] -= learning_rate * v_hat_w / (np.sqrt(s_hat_w) + epsilon)
            biases[i] -= learning_rate * v_hat_b / (np.sqrt(s_hat_b) + epsilon)
    return weights, biases

def train_model():
    # log config also so that i will be able to see the hyperparameters in it
    wandb.init()
    config = wandb.config
    wandb.run.name = f"hl_{config.hidden_sizes}_bs_{config.batch_size}_ac_{config.activation}"

    layers = [784] + config.hidden_sizes + [10]

    # Weight Initialization
    if config.weight_init == "random":
        weights = [np.random.randn(layers[i], layers[i+1]) * 0.01 for i in range(len(layers)-1)]
    elif config.weight_init == "xavier":
        weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / (layers[i] + layers[i+1])) for i in range(len(layers)-1)]
    
    biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
    
    opt_states = ([np.zeros_like(W) for W in weights], [np.zeros_like(b) for b in biases], 
                  [np.zeros_like(W) for W in weights], [np.zeros_like(b) for b in biases])
    
    batch_size = config.batch_size
    num_batches = train_images.shape[0] // batch_size

    for epoch in range(config.epochs):
        total_train_loss, total_train_acc = 0, 0

        for i in range(num_batches):
            X_batch = train_images[i * batch_size:(i + 1) * batch_size]
            y_batch = train_labels[i * batch_size:(i + 1) * batch_size]
            
            activations, hs = forward_pass(X_batch, weights, biases, config.activation)
            train_loss = cross_entropy_loss(y_batch, activations[-1])
            train_acc = np.mean(np.argmax(activations[-1], axis=1) == np.argmax(y_batch, axis=1))
            
            grads_W, grads_B = backward_pass(activations, hs, weights, y_batch, config.activation)
            
            weights, biases = update_weights(weights, biases, grads_W, grads_B, config, opt_states, epoch)
            
            total_train_loss += train_loss
            total_train_acc += train_acc

        # Average train loss & accuracy over batches
        avg_train_loss = total_train_loss / num_batches
        avg_train_acc = total_train_acc / num_batches

        # Compute validation loss and accuracy
        val_activations, _ = forward_pass(val_images, weights, biases, config.activation)
        val_loss = cross_entropy_loss(val_labels, val_activations[-1])
        val_acc = np.mean(np.argmax(val_activations[-1], axis=1) == np.argmax(val_labels, axis=1))

        # Log metrics in wandb
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    wandb.finish()

wandb.agent(sweep_id, function=train_model, count=10)

