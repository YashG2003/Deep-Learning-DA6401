import wandb
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset
from model import FlexibleCNN
from data_module import INaturalistDataModule
from config import best_config

def train_and_save_model(data_module=None):
    # Initialize wandb
    wandb.init(project='da6401_a2', name="best_cnn_model_run", config=best_config)
    wandb_logger = WandbLogger()

    # Create data module
    if data_module is None:
        data_module = INaturalistDataModule(
            train_path="inaturalist_12K/train",
            test_path="inaturalist_12K/val",
            batch_size=best_config['batch_size'],
            augment=best_config['data_augment'],
            img_size=(224, 224)
        )
        data_module.setup()

    # Create model
    model = FlexibleCNN(
        num_classes=10,
        filter_sizes=best_config['filter_sizes'],
        conv_activation=best_config['conv_activation'],
        dense_activation=best_config['dense_activation'],
        dense_neurons=best_config['dense_neurons'],
        kernel_size=best_config['kernel_size'],
        dropout_rate=best_config['dropout_rate'],
        weight_decay=best_config['weight_decay'],
        batch_norm=best_config['batch_norm'],
        learning_rate=best_config['learning_rate']
    )

    trainer = Trainer(
        max_epochs=20,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices = 1,      
        logger=wandb_logger,
        enable_progress_bar=True,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
        ],
        precision="16-mixed" if torch.cuda.is_available() else "bf16-mixed"
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Evaluate on test data
    trainer.test(model, datamodule=data_module)

    # Save the model weights
    torch.save(model.state_dict(), 'best_cnn_model_weights.pth')
    print("Model saved successfully")

    wandb.finish()

    return data_module

def create_prediction_grid(predictions, class_names):
    # Organize predictions class-wise
    class_samples = defaultdict(list)  # class_id -> list of (image, true_label, pred_label)

    for batch in predictions:
        for i in range(len(batch['images'])):
            img = batch['images'][i]
            true = batch['labels'][i].item()
            pred = batch['preds'][i].item()
            class_samples[true].append((img, true, pred))
    
    # Select 3 random samples per class
    samples = []
    for cls in range(len(class_names)):
        if len(class_samples[cls]) >= 3:
            selected = random.sample(class_samples[cls], 3)
        else:
            selected = class_samples[cls]  # Use all if less than 3
        samples.extend(selected)

    # Create figure
    plt.figure(figsize=(10, 20), constrained_layout=True)

    plt.figtext(0.5, 0.965, 
               f"Prediction on Test Samples",
               ha='center', fontsize=11)

    grid = plt.GridSpec(10, 3, top=0.94, bottom=0.02, hspace=0.35, wspace=0.10)

    for i, (image, true_label, pred_label) in enumerate(samples):
        ax = plt.subplot(grid[i])
        
        # Process image
        image = image.permute(1, 2, 0).cpu().numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        
        correct = true_label == pred_label
        border_color = '#2ecc71' if correct else '#e74c3c'
        
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)
        
        ax.imshow(image)
        ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
                    color='#34495e', fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

        if correct:
            ax.text(0.9, 0.07, "✓", transform=ax.transAxes,
                   color='green', fontsize=14, ha='right',
                   bbox=dict(facecolor='white', alpha=0.7))
        else:
            ax.text(0.9, 0.07, "✗", transform=ax.transAxes,
                   color='red', fontsize=14, ha='right',
                   bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.015, 1, 0.94])
    plt.savefig("class_balanced_predictions.png", dpi=150)
    plt.show()

def load_and_plot_model(data_module=None, model_path='best_cnn_model_weights.pth'):
    if data_module is None:
        data_module = INaturalistDataModule(
            train_path="inaturalist_12K/train",
            test_path="inaturalist_12K/val",
            batch_size=32,
            augment=False,
            img_size=(224, 224)
        )
        data_module.setup()
    
    # Load model
    model = FlexibleCNN(
        num_classes=10,
        filter_sizes=best_config['filter_sizes'],
        conv_activation=best_config['conv_activation'],
        dense_activation=best_config['dense_activation'],
        dense_neurons=best_config['dense_neurons'],
        kernel_size=best_config['kernel_size'],
        dropout_rate=best_config['dropout_rate'],
        weight_decay=best_config['weight_decay'],
        batch_norm=best_config['batch_norm'],
        learning_rate=best_config['learning_rate']
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Get test dataset
    test_dataset = data_module.test_dataset
    num_classes = len(test_dataset.classes)
    
    # Initialize dictionary with all class indices
    class_samples = {i: [] for i in range(num_classes)}
    
    # Collect exactly 3 samples per class
    for idx in range(len(test_dataset)):
        image, label = test_dataset[idx]
        if len(class_samples[label]) < 3:
            class_samples[label].append(image)
        
        # Early exit if we have all samples
        if all(len(samples) == 3 for samples in class_samples.values()):
            break
    
    # Prepare batch for prediction (sorted by class)
    images = []
    labels = []
    for class_idx in sorted(class_samples.keys()):
        images.extend(class_samples[class_idx])
        labels.extend([class_idx] * 3)
    
    # Create DataLoader
    custom_dataset = TensorDataset(torch.stack(images), torch.tensor(labels))
    custom_loader = DataLoader(custom_dataset, batch_size=30)
    
    # Get predictions
    trainer = Trainer(accelerator = "gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=False)
    predictions = trainer.predict(model, dataloaders=custom_loader)

    results = trainer.test(model, datamodule=data_module)
    print("The test accuracy is: ", results)
    
    # Create plot
    create_prediction_grid(predictions, test_dataset.classes)

if __name__ == "__main__":
    # 1. First train and save the model
    data_module = train_and_save_model()
    
    # 2. Later load and plot (can run multiple times)
    load_and_plot_model(data_module)