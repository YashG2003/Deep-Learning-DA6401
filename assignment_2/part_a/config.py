import torch

class Config:
    # Data
    train_path = "inaturalist_12K/train"    # local path of iNaturalist train dataset
    test_path = "inaturalist_12K/val"       # local path of iNaturalist test dataset
    img_size = (224, 224)
    batch_size = 32
    val_split = 0.2
    augment = True
    
    # Model
    num_classes = 10
    filter_sizes = [32, 64, 128, 256, 512]
    conv_activation = 'gelu'
    dense_activation = 'gelu'
    dense_neurons = 128
    kernel_size = 5
    dropout_rate = 0.2
    weight_decay = 0.001
    batch_norm = True
    learning_rate = 1e-4
    
    # Training
    max_epochs = 20
    precision = '16-mixed'
    limit_train_batches = 0.75
    
    # Misc
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Best hyperparameters configuration
best_config = {
    'batch_size': 32,
    'filter_sizes': [32, 64, 128, 256, 512],
    'conv_activation': 'gelu',
    'dense_activation': 'gelu',
    'dense_neurons': 128,
    'kernel_size': 5,
    'dropout_rate': 0.2,
    'weight_decay': 1e-3,
    'batch_norm': True,
    'data_augment': True,
    'learning_rate': 1e-4
}

# Sweep configuration
sweep_config = {
    'method': 'bayes',  
    'name': 'cnn_tuning_5',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'batch_size': {'values': [32, 64]},
        'filter_sizes': {'values': [[32,64,128,256,512], [32,64,64,128,128]]},
        'conv_activation': {'values': ['gelu']},
        'dense_activation': {'values': ['gelu']},
        'dense_neurons': {'values': [128, 256]},
        'kernel_size': {'values': [3, 5]},
        'dropout_rate': {'values': [0.2, 0.3]},
        'weight_decay': {'values': [1e-3, 1e-4]},
        'batch_norm': {'values': [True]},
        'data_augment': {'values': [True]},
        'learning_rate': {'values': [1e-3, 1e-4]}
    }
}

config = Config()