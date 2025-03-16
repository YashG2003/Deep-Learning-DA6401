import numpy as np
import os

class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        # Layer configuration
        if hasattr(config, 'hidden_layers'):
            # Use custom hidden layer sizes if provided (different hidden size in each layer)
            self.layers = [784] + config.hidden_layers + [10]       # config.hidden_layers = [[512, 256, 128]]
        else:
            # Default to original behavior
            self.layers = [784] + [config.hidden_size] * config.num_layers + [10]
        
        self.initialize_weights()
        self.initialize_optimizer_states()

        
    def initialize_weights(self):
        """Weight initialization based on config"""
        if self.config.weight_init == "random":
            self.weights = [np.random.randn(in_size, out_size) * 0.01 
                          for in_size, out_size in zip(self.layers[:-1], self.layers[1:])]
        elif self.config.weight_init == "xavier":
            self.weights = [np.random.randn(in_size, out_size) * np.sqrt(2 / (in_size + out_size))
                          for in_size, out_size in zip(self.layers[:-1], self.layers[1:])]
            
        self.biases = [np.zeros((1, out_size)) for out_size in self.layers[1:]]
    
    def initialize_optimizer_states(self):
        """Initialize optimizer momentum and RMS states"""
        self.momentum_w = [np.zeros_like(W) for W in self.weights]
        self.momentum_b = [np.zeros_like(b) for b in self.biases]
        self.squared_w = [np.zeros_like(W) for W in self.weights]
        self.squared_b = [np.zeros_like(b) for b in self.biases]
    
    @staticmethod
    def activation(x, name):
        """Activation function implementation"""
        if name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif name == "tanh":
            return np.tanh(x)
        elif name == "relu":
            return np.maximum(0, x)
        return x
    
    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.pre_activations = []
        
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.dot(self.activations[-1], W) + b
            self.pre_activations.append(h)
            self.activations.append(self.activation(h, self.config.activation))
            
        # Output layer (softmax)
        h_final = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.activations.append(self.softmax(h_final))
        return self.activations[-1]
    
    def backward(self, y_true):
        """Backward pass through the network"""
        grads_W = []
        grads_B = []
        dH = self.activations[-1] - y_true           # Gradient of cross-entropy loss w.r.t. softmax output
        #dH = 2*(activations[-1] - y_true)           # Gradient of mean squared error loss w.r.t. softmax output        
        
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            dW = np.dot(self.activations[i].T, dH) / y_true.shape[0]
            dB = np.sum(dH, axis=0, keepdims=True) / y_true.shape[0]
            
            # Add weight decay (L2 regularization) to the gradients
            if hasattr(self.config, 'weight_decay'):
                dW += self.config.weight_decay * self.weights[i]
            
            grads_W.insert(0, dW)
            grads_B.insert(0, dB)
            
            if i > 0:  # No activation gradient for input layer
                # Compute activation derivative
                if self.config.activation == "sigmoid":
                    d_activation = self.activations[i] * (1 - self.activations[i])
                elif self.config.activation == "tanh":
                    d_activation = 1 - self.activations[i]**2
                elif self.config.activation == "relu":
                    d_activation = (self.pre_activations[i-1] > 0).astype(float)
                
                dH = np.dot(dH, self.weights[i].T) * d_activation
                
        return grads_W, grads_B
    
    def update_parameters(self, grads_W, grads_B, epoch, X_batch, y_batch):
        """Update weights using specified optimizer"""
        lr = self.config.learning_rate
        opt = self.config.optimizer
        momentum = self.config.momentum
        beta = self.config.beta
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        epsilon = self.config.epsilon

        for i in range(len(self.weights)):
            if opt == "sgd":
                self.weights[i] -= lr * grads_W[i]
                self.biases[i] -= lr * grads_B[i]

            elif opt == "momentum":
                self.momentum_w[i] = momentum * self.momentum_w[i] + grads_W[i]
                self.momentum_b[i] = momentum * self.momentum_b[i] + grads_B[i]
                self.weights[i] -= lr * self.momentum_w[i]
                self.biases[i] -= lr * self.momentum_b[i]

            elif opt == "nag":
                # Create lookahead parameters
                lookahead_weights = [w.copy() for w in self.weights]
                lookahead_biases = [b.copy() for b in self.biases]
                lookahead_weights[i] = self.weights[i] - momentum * self.momentum_w[i]
                lookahead_biases[i] = self.biases[i] - momentum * self.momentum_b[i]

                # Temporary forward pass with lookahead parameters
                temp_model = NeuralNetwork(self.config)
                temp_model.weights = lookahead_weights
                temp_model.biases = lookahead_biases
                temp_output = temp_model.forward(X_batch)
                
                # Temporary backward pass
                temp_model.activations = temp_model.activations.copy()  # Prevent side effects
                temp_model.pre_activations = temp_model.pre_activations.copy()
                grads_lookahead_W, grads_lookahead_B = temp_model.backward(y_batch)

                # Update momentum buffers
                self.momentum_w[i] = momentum * self.momentum_w[i] + lr * grads_lookahead_W[i]
                self.momentum_b[i] = momentum * self.momentum_b[i] + lr * grads_lookahead_B[i]

                # Apply final update
                self.weights[i] -= self.momentum_w[i]
                self.biases[i] -= self.momentum_b[i]

            elif opt == "rmsprop":
                self.squared_w[i] = beta * self.squared_w[i] + (1 - beta) * grads_W[i]**2
                self.squared_b[i] = beta * self.squared_b[i] + (1 - beta) * grads_B[i]**2
                self.weights[i] -= lr * grads_W[i] / (np.sqrt(self.squared_w[i] + epsilon))
                self.biases[i] -= lr * grads_B[i] / (np.sqrt(self.squared_b[i] + epsilon))

            elif opt == "adam":
                self.momentum_w[i] = beta1 * self.momentum_w[i] + (1 - beta1) * grads_W[i]
                self.momentum_b[i] = beta1 * self.momentum_b[i] + (1 - beta1) * grads_B[i]
                self.squared_w[i] = beta2 * self.squared_w[i] + (1 - beta2) * grads_W[i]**2
                self.squared_b[i] = beta2 * self.squared_b[i] + (1 - beta2) * grads_B[i]**2

                v_hat_w = self.momentum_w[i] / (1 - beta1**(epoch+1))
                v_hat_b = self.momentum_b[i] / (1 - beta1**(epoch+1))
                s_hat_w = self.squared_w[i] / (1 - beta2**(epoch+1))
                s_hat_b = self.squared_b[i] / (1 - beta2**(epoch+1))

                self.weights[i] -= lr * v_hat_w / (np.sqrt(s_hat_w) + epsilon)
                self.biases[i] -= lr * v_hat_b / (np.sqrt(s_hat_b) + epsilon)

            elif opt == "nadam":
                self.momentum_w[i] = beta1 * self.momentum_w[i] + (1 - beta1) * grads_W[i]
                self.momentum_b[i] = beta1 * self.momentum_b[i] + (1 - beta1) * grads_B[i]
                self.squared_w[i] = beta2 * self.squared_w[i] + (1 - beta2) * grads_W[i]**2
                self.squared_b[i] = beta2 * self.squared_b[i] + (1 - beta2) * grads_B[i]**2

                v_hat_w = self.momentum_w[i] / (1 - beta1**(epoch+1))
                v_hat_b = self.momentum_b[i] / (1 - beta1**(epoch+1))
                s_hat_w = self.squared_w[i] / (1 - beta2**(epoch+1))
                s_hat_b = self.squared_b[i] / (1 - beta2**(epoch+1))

                momentum_corrected_w = beta1 * v_hat_w + (1 - beta1) * grads_W[i] / (1 - beta1**(epoch+1))
                momentum_corrected_b = beta1 * v_hat_b + (1 - beta1) * grads_B[i] / (1 - beta1**(epoch+1))

                self.weights[i] -= lr * momentum_corrected_w / (np.sqrt(s_hat_w) + epsilon)
                self.biases[i] -= lr * momentum_corrected_b / (np.sqrt(s_hat_b) + epsilon)

    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
    
    @staticmethod
    def mean_squared_error_loss(y_true, y_pred):
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
    
    def evaluate(self, X, y):
        """Evaluate model on given data"""
        outputs = self.forward(X)
        loss = self.cross_entropy_loss(y, outputs)
        acc = np.mean(np.argmax(outputs, axis=1) == np.argmax(y, axis=1))
        return loss, acc


def load_splitted_data():
    """Load pre-split dataset with verification"""
    files = [
        "train_images.npy", "train_labels.npy",
        "val_images.npy", "val_labels.npy",
        "test_images.npy", "test_labels.npy"
    ]
    
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Dataset file {f} not found. Run split_data.py first!")
    
    return (
        np.load("train_images.npy"),
        np.load("train_labels.npy"),
        np.load("val_images.npy"),
        np.load("val_labels.npy"),
        np.load("test_images.npy"),
        np.load("test_labels.npy")
    )

