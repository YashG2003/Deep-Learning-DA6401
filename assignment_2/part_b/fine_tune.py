import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import wandb
from model import FineTunedResNet
from data_module import INaturalistDataModule

def train_finetuned_model():
    # Initialize wandb
    wandb.init(project='da6401_a2', name="fine_tuning_resnet50")
    wandb_logger = WandbLogger()
    
    # Initialize data module
    data_module = INaturalistDataModule(
        train_path="inaturalist_12K/train",
        test_path="inaturalist_12K/val",
        batch_size=32,
        augment=True,
        img_size=(224, 224))
    
    data_module.setup()
    
    # Create model
    model = FineTunedResNet(num_classes=10, learning_rate=1e-4)
    
    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "bf16-mixed",
        callbacks=[
            ModelCheckpoint(monitor='val_acc', mode='max')
        ]
    )
    
    # Train and test
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    wandb.finish()
    return model

if __name__ == "__main__":
    model = train_finetuned_model()