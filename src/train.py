import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
import argparse

from model import VisionTransformer, vit_tiny_patch16_224, vit_small_patch16_224
import json
from datetime import datetime


class WarmupCosineSchedule:
    """
    Learning rate scheduler with warmup and cosine annealing.
    Critical for ViT training - from original paper.
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

def unpickle(file):
    """Load CIFAR-100 batch file."""
    import pickle
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
    return dicts


class CIFAR100Dataset(Dataset):
    """
    Custom CIFAR-100 Dataset that loads from pickle files.
    
    Dataset format:
    - data: 10000x3072 array (32x32x3 images, row-major order)
    - labels: list of 10000 integers (0-99)
    """
    
    def __init__(self, data_dir, train=True, transform=None):
        """
        Args:
            data_dir: Path to CIFAR-100 data directory
            train: If True, load training batches; if False, load test batch
            transform: Optional transform to apply to images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        
        # Load data and labels
        self.data = []
        self.labels = []
        7


        if train:
            batch_file = os.path.join(data_dir, 'train')
            batch_dict = unpickle(batch_file)
            self.data.append(batch_dict[b'data'])
            self.labels.extend(batch_dict[b'fine_labels'])
            self.data = np.vstack(self.data)  # Shape: (50000, 3072)
        else:
            # Load test batch
            test_file = os.path.join(data_dir, 'test')
            test_dict = unpickle(test_file)
            self.data = test_dict[b'data']  # Shape: (10000, 3072)
            self.labels = test_dict[b'fine_labels']
        
        # Reshape data from (N, 3072) to (N, 3, 32, 32)
        self.data = self.data.reshape(-1, 3, 32, 32)
        # Keep as uint8 for RandAugment (don't convert to float yet!)
        # Transforms will handle the conversion later
        
        # Load label names
        meta_file = os.path.join(data_dir, 'meta')
        meta_dict = unpickle(meta_file)
        self.label_names = [name.decode('utf-8') for name in meta_dict[b'fine_label_names']]
        
        print(f"Loaded {'training' if train else 'test'} data: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image as uint8 for RandAugment compatibility
        image = torch.from_numpy(self.data[idx].copy())  # Shape: (3, 32, 32), uint8
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, train=True):
    """
    Get data transforms for CIFAR-100.
    
    Args:
        img_size: Target image size (ViT typically uses 224x224)
        train: If True, include data augmentation
    
    Note:
        Enhanced augmentation based on ViT paper recommendations:
        - RandAugment for better regularization
        - Random Erasing (Cutout) for robustness
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),        # Augment at original size
            transforms.RandomHorizontalFlip(p=0.5),
            # Paper recommendation: RandAugment for better augmentation
            transforms.RandAugment(num_ops=2, magnitude=9),
            # Convert to float [0, 1] after augmentation
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((img_size, img_size)),     # Then resize to target
            transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],  # CIFAR-100 mean
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]    # CIFAR-100 std
            ),
            # Paper recommendation: Random Erasing (Cutout)
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
        ])
    else:
        return transforms.Compose([
            # Convert to float [0, 1] for consistency
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            ),
        ])


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scheduler=None, grad_clip=1.0):
    """Train for one epoch with gradient clipping and per-step scheduler."""
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
        
        # Backward pass with gradient clipping (paper requirement)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        
        # Update scheduler per step (for warmup)
        if scheduler is not None:
            scheduler.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': f'{current_lr:.6f}'
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

class TrainingHistory: 
    
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
        self.timestamps = []
    
    def add_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
        self.timestamps.append(datetime.now().isoformat())
    
    def save(self, filepath):
        history = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'timestamps': self.timestamps
        }
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load(self, filepath):
        with open(filepath, 'r') as f:
            history = json.load(f)
        self.epochs = history['epochs']
        self.train_losses = history['train_losses']
        self.train_accs = history['train_accs']
        self.val_losses = history['val_losses']
        self.val_accs = history['val_accs']
        self.learning_rates = history['learning_rates']
        self.timestamps = history['timestamps']
     
    def get_summary(self):
        if not self.epochs:
            return'No recorded history.'
        
        return {
            'total_epochs': len(self.epochs),
            'best_val_accuracy': max(self.val_accs) if self.val_accs else 0,
            'final_train_accuracy': self.train_accs[-1] if self.train_accs else 0,
            'final_val_accuracy': self.val_accs[-1] if self.val_accs else 0,
            'overfitting_gap': (self.train_accs[-1] - self.val_accs[-1]) if self.train_accs and self.val_accs else 0
        }

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print(f"\nLoading data from: {args.data_dir}")
    train_dataset = CIFAR100Dataset(
        args.data_dir, 
        train=True, 
        transform=get_transforms(args.img_size, train=True)
    )
    val_dataset = CIFAR100Dataset(
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
        model = vit_tiny_patch16_224(num_classes=100)
    elif args.model == 'small':
        model = vit_small_patch16_224(num_classes=100)
    elif args.model == 'custom':
        # Custom ViT for smaller images
        model = VisionTransformer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=3,
            num_classes=100,
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
    
    # Learning rate scheduler with warmup (paper requirement)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_epochs * len(train_loader)
    
    print(f"\nScheduler settings:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({args.warmup_epochs} epochs)")
    print(f"  Base LR: {args.lr}")
    print(f"  Gradient clipping: {args.grad_clip}")
    
    if args.scheduler == 'warmup_cosine':
        # Use warmup + cosine annealing (paper recommendation)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-6
        )
    elif args.scheduler == 'cosine':
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

    history = TrainingHistory()
    
    # Training loop
    best_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train with scheduler and gradient clipping
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scheduler=scheduler if args.scheduler == 'warmup_cosine' else None,
            grad_clip=args.grad_clip
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 添加到历史记录
        history.add_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
       
        # Update learning rate (only for epoch-based schedulers, warmup_cosine updates per step)
        if scheduler and args.scheduler != 'warmup_cosine':
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
                'args': args,
                'training_history': {  # 在检查点中保存历史
                    'epochs': history.epochs,
                    'train_losses': history.train_losses,
                    'train_accs': history.train_accs,
                    'val_losses': history.val_losses,
                    'val_accs': history.val_accs,
                    'learning_rates': history.learning_rates
                }
            }
            
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(checkpoint, save_path)
             # 同时保存训练历史到JSON文件
            history.save(os.path.join(args.save_dir, 'training_history.json'))
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args,
                'training_history': {  # 在定期检查点中也保存历史
                    'epochs': history.epochs,
                    'train_losses': history.train_losses,
                    'train_accs': history.train_accs,
                    'val_losses': history.val_losses,
                    'val_accs': history.val_accs,
                    'learning_rates': history.learning_rates
                }
            }
            
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"  Saved checkpoint: {save_path}")
    # 训练结束后保存完整历史
    history.save(os.path.join(args.save_dir, 'final_training_history.json'))
    
    print("\n" + "=" * 80)
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    print("=" * 80)

    return history  # 返回历史记录对象

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR-100')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../cifar-100-python/cifar-100-python',
                        help='Path to CIFAR-100 data directory (default: ../cifar-100-python/cifar-100-python)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    
    # Model parameters (use 'small' for CIFAR-100 due to 100 classes)
    parser.add_argument('--model', type=str, default='small',
                        choices=['tiny', 'small', 'custom'],
                        help='Model variant (default: small for CIFAR-100)')
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
    
    # Training parameters (updated based on ViT paper for CIFAR-10)
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 300, paper uses 300-1000)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3, adjusted for CIFAR-100 from scratch)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay (default: 0.1, paper uses 0.1)')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                        choices=['warmup_cosine', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler (default: warmup_cosine)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs (default: 10, paper recommendation)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default: 1.0, paper uses 1.0)')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default: 0, use 0 for macOS)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    main(args)