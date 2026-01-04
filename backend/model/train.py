"""
Training script for the Deepfake Detection model.
Usage: python -m model.train
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import get_model
from model.dataset import get_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


def train(args):
    """Main training function."""
    print("=" * 60)
    print("Deepfake Detection Model Training")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found: {data_dir}")
        print("Please create the following structure:")
        print("  data/")
        print("    train/")
        print("      real/   <- Put real images here")
        print("      fake/   <- Put fake/manipulated images here")
        print("    val/")
        print("      real/   <- Put validation real images here")
        print("      fake/   <- Put validation fake images here")
        return
    
    # Load data
    print(f"\nLoading data from: {data_dir}")
    train_loader, val_loader = get_dataloaders(
        data_dir, 
        batch_size=args.batch_size,
        num_workers=0
    )
    
    if len(train_loader.dataset) == 0:
        print("\nERROR: No training data found!")
        print("Please add images to data/train/real/ and data/train/fake/")
        return
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, 'deepfake_detector.pth')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        if len(val_loader.dataset) > 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), weights_path)
                print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            # No validation data, save based on training accuracy
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                torch.save(model.state_dict(), weights_path)
                print(f"  ✓ Saved model (Train Acc: {train_acc:.2f}%)")
        
        scheduler.step()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {weights_path}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test-run', action='store_true', help='Quick test run')
    
    args = parser.parse_args()
    
    if args.test_run:
        args.epochs = 2
        args.batch_size = 4
    
    train(args)
