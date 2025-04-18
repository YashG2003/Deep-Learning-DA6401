import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

class INaturalistDataModule(pl.LightningDataModule):   
    _log_hyperparams: bool = True 
    allow_zero_length_dataloader_with_multiple_devices: bool = False
    
    def __init__(self, train_path, test_path, batch_size=32, val_split=0.2, 
                 img_size=(224, 224), augment=False):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.img_size = img_size
        self.augment = augment
        
    def setup(self, stage=None):
        # Define transformations
        base_transform = [
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if self.augment:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                *base_transform
            ])
        else:
            train_transform = transforms.Compose(base_transform)
            
        test_transform = transforms.Compose(base_transform)
        
        # Load full dataset
        full_dataset = ImageFolder(self.train_path, transform=train_transform)
        
        # Create class-wise indices
        class_indices = {}
        for idx, (_, label) in enumerate(full_dataset.samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Split each class into train and val
        train_idx = []
        val_idx = []
        for label, indices in class_indices.items():
            np.random.shuffle(indices)
            split = int(len(indices) * self.val_split)
            val_idx.extend(indices[:split])
            train_idx.extend(indices[split:])
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        
        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset = Subset(full_dataset, val_idx)
        self.test_dataset = ImageFolder(self.test_path, transform=test_transform)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=1,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=1,
            drop_last=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=1,
            drop_last=True
        )