"""
Fine-tune CIFAR-100 trained model on CIFAR-10
Option A: Unfreeze last 4 transformer blocks for better adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import pickle

from model import vit_small_patch16_224
from torchvision import transforms


class CIFAR10Dataset(torch.utils.data.Dataset):
    """CIFAR-10 Dataset"""
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        
        if train:
            # Load training batches
            for i in range(1, 6):
                file_path = os.path.join(data_dir, f'data_batch_{i}')
                with open(file_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    self.data.append(batch[b'data'])
                    self.labels.extend(batch[b'labels'])
            self.data = np.concatenate(self.data)
        else:
            # Load test batch
            file_path = os.path.join(data_dir, 'test_batch')
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data = batch[b'data']
                self.labels = batch[b'labels']
        
        # Reshape data: (N, 3072) -> (N, 3, 32, 32)
        self.data = self.data.reshape(-1, 3, 32, 32)
        
        print(f"Loaded {'train' if train else 'test'} data: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx]).float()
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, train=True):
    """
    Get transforms for CIFAR-10 with CIFAR-100 normalization
    (backbone was trained on CIFAR-100)
    """
    # Use CIFAR-100 normalization (backbone expects this!)
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=mean, std=std),
        ])


def load_cifar100_backbone(checkpoint_path, device):
    """Load CIFAR-100 trained model and prepare for fine-tuning"""
    print(f"Loading CIFAR-100 pretrained model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create ViT-Small with CIFAR-100 classes first
    model_cifar100 = vit_small_patch16_224(num_classes=100)
    model_cifar100.load_state_dict(checkpoint['model_state_dict'])
    
    # Create new model with CIFAR-10 classes
    model_cifar10 = vit_small_patch16_224(num_classes=10)
    
    # Copy all weights except the classification head
    with torch.no_grad():
        # Copy patch embedding
        model_cifar10.patch_embed.load_state_dict(model_cifar100.patch_embed.state_dict())
        
        # Copy CLS token and position embeddings
        model_cifar10.cls_token.copy_(model_cifar100.cls_token)
        model_cifar10.pos_embed.copy_(model_cifar100.pos_embed)
        
        # Copy all transformer blocks
        for i in range(len(model_cifar10.blocks)):
            model_cifar10.blocks[i].load_state_dict(model_cifar100.blocks[i].state_dict())
        
        # Copy layer norm
        model_cifar10.norm.load_state_dict(model_cifar100.norm.state_dict())
        
        # Classification head will be randomly initialized (different output size)
    
    print(f"[OK] Loaded pretrained backbone from CIFAR-100 model")
    print(f"[OK] Classification head initialized for 10 classes")
    
    return model_cifar10.to(device)


def freeze_layers(model, unfreeze_last_n_blocks=4):
    """
    Freeze all layers except:
    - Classification head
    - Last N transformer blocks
    """
    print(f"\nFreezing strategy:")
    print(f"  - Freeze: Patch embedding, position embeddings, CLS token")
    print(f"  - Freeze: First {12 - unfreeze_last_n_blocks} transformer blocks")
    print(f"  - Unfreeze: Last {unfreeze_last_n_blocks} transformer blocks")
    print(f"  - Unfreeze: Classification head")
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last N transformer blocks
    for i in range(12 - unfreeze_last_n_blocks, 12):
        print(f"  [Unfreezing] Transformer Block {i+1}")
        for param in model.blocks[i].parameters():
            param.requires_grad = True
    
    # Unfreeze final layer norm
    print(f"  [Unfreezing] Final LayerNorm")
    for param in model.norm.parameters():
        param.requires_grad = True
    
    # Unfreeze classification head
    print(f"  [Unfreezing] Classification Head")
    for param in model.head.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
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
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(test_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main():
    # Configuration
    print("="*80)
    print("CIFAR-10 FINE-TUNING (Option A: Unfreeze Last 4 Blocks)")
    print("="*80)
    
    # Paths
    pretrained_checkpoint = './checkpoints/best_model.pth'  # CIFAR-100 trained model
    data_dir = '../cifar-10-python/cifar-10-batches-py'
    save_dir = './cifar10_finetuned_optionA'
    
    # Hyperparameters
    batch_size = 128
    epochs = 50  # Fine-tuning typically needs fewer epochs
    learning_rate = 1e-4  # Lower LR for fine-tuning
    weight_decay = 0.01
    unfreeze_last_n_blocks = 4
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Pretrained checkpoint: {pretrained_checkpoint}")
    print(f"Fine-tuning epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Unfreezing last {unfreeze_last_n_blocks} transformer blocks")
    print()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load datasets
    print("Loading CIFAR-10 dataset...")
    train_dataset = CIFAR10Dataset(
        data_dir, 
        train=True, 
        transform=get_transforms(img_size=224, train=True)
    )
    test_dataset = CIFAR10Dataset(
        data_dir, 
        train=False, 
        transform=get_transforms(img_size=224, train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load pretrained model
    print("\n" + "="*80)
    print("LOADING PRETRAINED MODEL")
    print("="*80)
    model = load_cifar100_backbone(pretrained_checkpoint, device)
    
    # Freeze layers (Option A strategy)
    print("\n" + "="*80)
    print("FREEZING STRATEGY")
    print("="*80)
    model = freeze_layers(model, unfreeze_last_n_blocks=unfreeze_last_n_blocks)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize unfrozen parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  [OK] New best accuracy: {best_acc:.2f}%")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'config': {
                    'pretrained_on': 'CIFAR-100',
                    'finetuned_on': 'CIFAR-10',
                    'unfrozen_blocks': unfreeze_last_n_blocks,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                }
            }
            
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, checkpoint_path)
            print(f"  [OK] Saved checkpoint: {checkpoint_path}")
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_dir}/best_model.pth")
    print(f"Training history saved to: {history_path}")
    print("="*80)


if __name__ == '__main__':
    main()

