"""
Dataset loader for Deepfake Detection training.
Loads images from train/real and train/fake directories.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class DeepfakeDataset(Dataset):
    """Dataset for loading real/fake images for deepfake detection."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to data directory containing train/val subdirs
            split: 'train' or 'val'
            transform: Optional transform to apply to images
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        
        # Load real images (label 0)
        real_dir = os.path.join(self.root_dir, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append((os.path.join(real_dir, img_name), 0))
        
        # Load fake images (label 1)
        fake_dir = os.path.join(self.root_dir, 'fake')
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append((os.path.join(fake_dir, img_name), 1))
        
        # Shuffle samples
        random.shuffle(self.samples)
        
        print(f"Loaded {len(self.samples)} images for {split} split")
        print(f"  - Real: {len([s for s in self.samples if s[1] == 0])}")
        print(f"  - Fake: {len([s for s in self.samples if s[1] == 1])}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train'):
    """Get transforms for training or validation."""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(data_dir, batch_size=32, num_workers=0):
    """Create train and validation dataloaders."""
    train_dataset = DeepfakeDataset(
        data_dir, 
        split='train', 
        transform=get_transforms('train')
    )
    val_dataset = DeepfakeDataset(
        data_dir, 
        split='val', 
        transform=get_transforms('val')
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
