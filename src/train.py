import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
import argparse
from pathlib import Path

from model import VisionTransformer, vit_tiny_patch16_224, vit_small_patch16_224


def unpickle(file):
    """Load CIFAR-10 batch file."""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR10Dataset(Dataset):
    """
    Custom CIFAR-10 Dataset that loads from pickle files.
    
    Dataset format:
    - data: 10000x3072 array (32x32x3 images, row-major order)
    - labels: list of 10000 integers (0-9)
    """
    
    def __init__(self, data_dir, train=True, transform=None):
        """
        Args:
            data_dir: Path to CIFAR-10 data directory
            train: If True, load training batches; if False, load test batch
            transform: Optional transform to apply to images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        
        # Load data and labels
        self.data = []
        self.labels = []
        
        if train:
            # Load all 5 training batches
            for i in range(1, 6):
                batch_file = os.path.join(data_dir, f'data_batch_{i}')
                batch_dict = unpickle(batch_file)
                self.data.append(batch_dict[b'data'])
                self.labels.extend(batch_dict[b'labels'])
            
            self.data = np.vstack(self.data)  # Shape: (50000, 3072)
        else:
            # Load test batch
            test_file = os.path.join(data_dir, 'test_batch')
            test_dict = unpickle(test_file)
            self.data = test_dict[b'data']  # Shape: (10000, 3072)
            self.labels = test_dict[b'labels']
        
        # Reshape data from (N, 3072) to (N, 3, 32, 32)
        self.data = self.data.reshape(-1, 3, 32, 32)
        # Convert from [0, 255] to [0, 1]
        self.data = self.data.astype(np.float32) / 255.0
        
        # Load label names
        meta_file = os.path.join(data_dir, 'batches.meta')
        meta_dict = unpickle(meta_file)
        self.label_names = [name.decode('utf-8') for name in meta_dict[b'label_names']]
        
        print(f"Loaded {'training' if train else 'test'} data: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx])  # Shape: (3, 32, 32)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, train=True):
    """
    Get data transforms for CIFAR-10.
    
    Args:
        img_size: Target image size (ViT typically uses 224x224)
        train: If True, include data augmentation
    
    Note:
        For training, augmentation is performed on 32x32 before resizing to preserve
        the relative scale of augmentation (padding=4 on 32x32 = 12.5% range).
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),        # Augment at original size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((img_size, img_size)),     # Then resize to target
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 mean
                std=[0.2470, 0.2435, 0.2616]    # CIFAR-10 std
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print(f"\nLoading data from: {args.data_dir}")
    train_dataset = CIFAR10Dataset(
        args.data_dir, 
        train=True, 
        transform=get_transforms(args.img_size, train=True)
    )
    val_dataset = CIFAR10Dataset(
        args.data_dir, 
        train=False, 
        transform=get_transforms(args.img_size, train=False)
    )
    
    # Create dataloaders
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda
    )
    
    # Create model
    print(f"\nCreating ViT model: {args.model}")
    if args.model == 'tiny':
        model = vit_tiny_patch16_224(num_classes=10)
    elif args.model == 'small':
        model = vit_small_patch16_224(num_classes=10)
    elif args.model == 'custom':
        # Custom ViT for smaller images
        model = VisionTransformer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=3,
            num_classes=10,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=4.0,
            dropout=args.dropout,
            embed_dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Training loop
    best_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 80)
        
        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  ✓ New best accuracy! Saving checkpoint...")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }
            
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(checkpoint, save_path)
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }
            
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"  Saved checkpoint: {save_path}")
    
    print("\n" + "=" * 80)
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR-10')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../cifar-10-batches-py',
                        help='Path to CIFAR-10 data directory (default: ../cifar-10-batches-py)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'small', 'custom'],
                        help='Model variant (default: tiny)')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size for custom model (default: 16)')
    parser.add_argument('--embed_dim', type=int, default=192,
                        help='Embedding dimension for custom model (default: 192)')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of transformer blocks for custom model (default: 12)')
    parser.add_argument('--num_heads', type=int, default=3,
                        help='Number of attention heads for custom model (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler (default: cosine)')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default: 0, use 0 for macOS)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    main(args)