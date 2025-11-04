"""
Utility functions for testing and evaluating the Vision Transformer model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from model import VisionTransformer, vit_tiny_patch16_224, vit_small_patch16_224
from train import CIFAR10Dataset, get_transforms


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[VisionTransformer, Dict]:
    """
    Load a trained Vision Transformer model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded Vision Transformer model
        checkpoint: Full checkpoint dictionary with training info
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration from checkpoint
    args = checkpoint['args']
    
    # Create model based on saved configuration
    if args.model == 'tiny':
        model = vit_tiny_patch16_224(num_classes=10)
    elif args.model == 'small':
        model = vit_small_patch16_224(num_classes=10)
    else:
        # Custom model
        model = VisionTransformer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            num_classes=10,
            dropout=args.dropout
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  - Trained for {checkpoint['epoch']} epochs")
    print(f"  - Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    print(f"  - Model variant: {args.model}")
    
    return model, checkpoint


def test_model(
    model: nn.Module,
    data_dir: str = '../cifar-10-python/cifar-10-batches-py',
    batch_size: int = 128,
    img_size: int = 224,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Test the model on CIFAR-10 test set.
    
    Args:
        model: The Vision Transformer model to test
        data_dir: Path to CIFAR-10 data directory
        batch_size: Batch size for testing
        img_size: Input image size
        device: Device to run testing on
        verbose: Whether to show progress bar
    
    Returns:
        results: Dictionary containing test metrics
    """
    # Create test dataset and loader
    test_dataset = CIFAR10Dataset(
        data_dir,
        train=False,
        transform=get_transforms(img_size, train=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    
    model.eval()
    
    # Track metrics
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    all_predictions = []
    all_labels = []
    
    print(f"\n{'='*80}")
    print(f"Testing model on {len(test_dataset)} test images...")
    print(f"{'='*80}")
    
    with torch.no_grad():
        iterator = tqdm(test_loader, desc="Testing", disable=not verbose)
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class statistics
            for label, prediction in zip(labels, predicted):
                all_labels.append(label.item())
                all_predictions.append(prediction.item())
                
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1
            
            # Update progress bar
            if verbose:
                current_acc = 100 * correct / total
                iterator.set_postfix({'Accuracy': f'{current_acc:.2f}%'})
    
    # Calculate overall accuracy
    overall_accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            class_accuracies[CIFAR10_CLASSES[i]] = acc
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TEST RESULTS")
    print(f"{'='*80}")
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"\nPer-Class Accuracy:")
    print(f"{'-'*80}")
    
    for class_name, acc in class_accuracies.items():
        bar_length = int(acc / 2)  # Scale to 50 chars max
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{class_name:>12}: {bar} {acc:6.2f}%")
    
    print(f"{'='*80}\n")
    
    # Prepare results dictionary
    results = {
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'class_accuracies': class_accuracies,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return results


def visualize_predictions(
    model: nn.Module,
    data_dir: str = '../cifar-10-python/cifar-10-batches-py',
    img_size: int = 224,
    device: str = 'cpu',
    num_samples: int = 16,
    save_path: str = None
):
    """
    Visualize model predictions on random test samples.
    
    Args:
        model: The Vision Transformer model
        data_dir: Path to CIFAR-10 data directory
        img_size: Input image size
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    # Create test dataset
    test_dataset = CIFAR10Dataset(
        data_dir,
        train=False,
        transform=get_transforms(img_size, train=False)
    )
    
    # Get random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    model.eval()
    
    # Create subplot grid
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            image, label = test_dataset[idx]
            
            # Make prediction
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            _, predicted = torch.max(output, 1)
            
            # Get prediction probabilities
            probs = torch.softmax(output, dim=1)
            confidence = probs[0, predicted].item() * 100
            
            # Denormalize image for display
            img_display = image.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2616])
            img_display = img_display * std + mean
            img_display = np.clip(img_display, 0, 1)
            
            # Display image
            ax.imshow(img_display)
            ax.axis('off')
            
            # Set title with color based on correctness
            pred_label = CIFAR10_CLASSES[predicted.item()]
            true_label = CIFAR10_CLASSES[label]
            
            if predicted.item() == label:
                title_color = 'green'
                title = f'✓ {pred_label}\n({confidence:.1f}%)'
            else:
                title_color = 'red'
                title = f'✗ Pred: {pred_label}\nTrue: {true_label}\n({confidence:.1f}%)'
            
            ax.set_title(title, color=title_color, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def confusion_matrix(
    predictions: List[int],
    labels: List[int],
    save_path: str = None
):
    """
    Create and display confusion matrix.
    
    Args:
        predictions: List of predicted class indices
        labels: List of true class indices
        save_path: Optional path to save the figure
    """
    from sklearn.metrics import confusion_matrix as cm
    import seaborn as sns
    
    # Compute confusion matrix
    conf_matrix = cm(labels, predictions)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - CIFAR-10 Test Set', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


# Main testing script
if __name__ == '__main__':
    # Configuration
    checkpoint_path = 'checkpoints/best_model.pth'
    data_dir = '../cifar-10-python/cifar-10-batches-py'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(checkpoint_path, device=device)
    
    # Test model
    results = test_model(
        model,
        data_dir=data_dir,
        batch_size=128,
        img_size=224,
        device=device,
        verbose=True
    )
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(
        model,
        data_dir=data_dir,
        img_size=224,
        device=device,
        num_samples=16,
        save_path='test_predictions.png'
    )
    
    # Show confusion matrix (requires scikit-learn and seaborn)
    try:
        print("\nGenerating confusion matrix...")
        confusion_matrix(
            results['predictions'],
            results['labels'],
            save_path='confusion_matrix.png'
        )
    except ImportError:
        print("Skipping confusion matrix (requires scikit-learn and seaborn)")
