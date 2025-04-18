import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW

class FlexibleCNN(pl.LightningModule):
    def __init__(self, num_classes=10, 
                 filter_sizes=[16,32,64,128,256],
                 conv_activation='relu', 
                 dense_activation='relu',
                 dense_neurons=128,
                 kernel_size=3,
                 dropout_rate=0.2,
                 weight_decay=0.01,
                 batch_norm=False,
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Dynamically create convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 3  # RGB input
        
        # Create 5 conv-activation-pool blocks
        for i, out_channels in enumerate(filter_sizes):
            last_block = (i == len(filter_sizes)-1)
            self.conv_blocks.append(self._create_conv_block(in_channels, out_channels, i, last_block))
            in_channels = out_channels
        
        # Calculate flattened size after conv blocks
        self.flattened_size = self._get_conv_output((3, 224, 224))
        
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(self.flattened_size, dense_neurons),
            nn.BatchNorm1d(dense_neurons) if batch_norm else nn.Identity(),
            self._get_activation(dense_activation),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_neurons, num_classes)
        )
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        
    def _create_conv_block(self, in_channels, out_channels, idx, last_block=False):
        block = nn.Sequential()
        
        # Conv layer
        block.add_module(f'conv_{idx}', nn.Conv2d(
            in_channels, out_channels,
            kernel_size=self.hparams.kernel_size,
            padding=self.hparams.kernel_size//2
        ))
        
        # Activation
        block.add_module(f'act_{idx}', self._get_activation(self.hparams.conv_activation))
        
        # Batch norm if enabled
        if self.hparams.batch_norm:
            block.add_module(f'bn_{idx}', nn.BatchNorm2d(out_channels))
        
        # Max pooling
        block.add_module(f'pool_{idx}', nn.MaxPool2d(2))
        
        # Dropout except last block
        if self.hparams.dropout_rate > 0 and not last_block:
            block.add_module(f'dropout_{idx}', nn.Dropout2d(self.hparams.dropout_rate))
        
        return block
        
    def _get_activation(self, name):
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'gelu':
            return nn.GELU()
        elif name.lower() == 'silu':
            return nn.SiLU()
        elif name.lower() == 'mish':
            return nn.Mish()
        else:
            raise ValueError(f"Unsupported activation: {name}")
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self._forward_conv(input)
            return int(np.prod(output.size()[1:]))
    
    def _forward_conv(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x
    
    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense(x)
        return x
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay  
        )
        
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate*1.5,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.25,
                div_factor=8,
                final_div_factor=50
            ),
            'interval': 'step'
        }
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        self.val_acc(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        self.test_acc(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {'images': x, 'labels': y, 'preds': preds}