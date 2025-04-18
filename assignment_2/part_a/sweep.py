import wandb
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from model import FlexibleCNN
from data_module import INaturalistDataModule
from config import sweep_config

def train_sweep():
    # Initialize wandb
    wandb.init()

    run_name = (
        f"lr_{wandb.config.learning_rate}_"
        f"ca_{wandb.config.conv_activation}_"
        f"ks_{wandb.config.kernel_size}_"
        f"fs_{wandb.config.filter_sizes}"
    )
    wandb.run.name = run_name
    
    # Create data module
    data_module = INaturalistDataModule(
        train_path="inaturalist_12K/train",
        test_path="inaturalist_12K/val",
        batch_size=wandb.config.batch_size,
        augment=wandb.config.data_augment,
        img_size=(224, 224)
    )
    
    # Create model
    model = FlexibleCNN(
        num_classes=10,
        filter_sizes=wandb.config.filter_sizes,
        conv_activation=wandb.config.conv_activation,
        dense_activation=wandb.config.dense_activation,
        dense_neurons=wandb.config.dense_neurons,
        kernel_size=wandb.config.kernel_size,
        dropout_rate=wandb.config.dropout_rate,
        weight_decay=wandb.config.weight_decay,
        batch_norm=wandb.config.batch_norm,
        learning_rate=wandb.config.learning_rate
    )
    
    trainer = Trainer(
        max_epochs=15,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices = 1,   
        logger=WandbLogger(),
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(monitor="val_acc", patience=5, mode="max", min_delta=0.005),
            ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
        ],
        precision="16-mixed" if torch.cuda.is_available() else "bf16-mixed"
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Test the model
    trainer.test(model, datamodule=data_module)
    
    wandb.finish()

def run_sweep():
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project='da6401_a2')
    
    # Run sweep
    wandb.agent(sweep_id, function=train_sweep, count=20)  # Run 20 trials

if __name__ == "__main__":
    run_sweep()