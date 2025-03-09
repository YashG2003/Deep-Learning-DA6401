import wandb
import numpy as np
from model import NeuralNetwork, load_data

def train_model():
    wandb.init()
    config = wandb.config
    wandb.run.name = (f"hl_{config.num_layers}_{config.hidden_size}_"
                     f"bs_{config.batch_size}_ac_{config.activation}")
    
    # Load data
    train_images, train_labels, val_images, val_labels, _, _ = load_data()
    
    # Initialize model
    model = NeuralNetwork(config)
    batch_size = config.batch_size
    num_batches = train_images.shape[0] // batch_size
    
    for epoch in range(config.epochs):
        train_loss, train_acc = 0, 0
        
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
            
            train_loss += batch_loss
            train_acc += batch_acc
        
        # Validation
        val_loss, val_acc = model.evaluate(val_images, val_labels)
        
        # Log metrics
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss/num_batches,
            "train_acc": train_acc/num_batches,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
    
    wandb.finish()

# Initialize wandb sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [15, 20]},
        "num_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [256, 128, 64]},
        #"hidden_layers": {"values": [[256, 128, 64]]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "momentum": {"values": [0.9, 0.95]},
        "beta": {"values": [0.9, 0.95]},
        "beta1": {"values": [0.9, 0.85]},
        "beta2": {"values": [0.999, 0.9]},
        "epsilon": {"values": [1e-8]},
        "batch_size": {"values": [32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["relu", "sigmoid", "tanh"]}
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-nn")
    wandb.agent(sweep_id, function=train_model, count=1)
