"""
Visualize attention maps from Vision Transformer to understand what the model learns.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import VisionTransformer
from train import CIFAR100Dataset, get_transforms
from utils import load_model, CIFAR100_CLASSES


def get_attention_maps(model, image, device='cpu'):
    """
    Extract attention maps from all transformer blocks by manually computing them.
    
    Args:
        model: Vision Transformer model
        image: Input image tensor (1, 3, H, W)
        device: Device to run on
    
    Returns:
        attention_maps: List of attention maps from each block
    """
    model.eval()
    attention_maps = []
    
    with torch.no_grad():
        x = image.to(device)
        B = x.shape[0]
        
        # Patch embedding
        x = model.patch_embed(x)
        
        # Add CLS token
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + model.pos_embed
        x = model.pos_dropout(x)
        
        # Go through each transformer block and manually compute attention
        for block in model.blocks:
            # Pre-norm
            x_norm = block.norm1(x)
            
            # Manually compute attention weights (replicate MultiHeadAttention forward)
            B, N, C = x_norm.shape
            attn_module = block.attn
            
            # Get QKV
            qkv = attn_module.qkv(x_norm).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention weights (before dropout)
            attn = (q @ k.transpose(-2, -1)) * attn_module.scale
            attn = attn.softmax(dim=-1)
            
            # Store attention weights
            attention_maps.append(attn.detach().cpu())
            
            # Complete the block forward (for next iteration)
            attn_dropped = attn_module.attn_dropout(attn)
            x_attn = (attn_dropped @ v).transpose(1, 2).reshape(B, N, C)
            x_attn = attn_module.proj(x_attn)
            x_attn = attn_module.proj_dropout(x_attn)
            
            # Residual connection
            x = x + x_attn
            # MLP block
            x = x + block.mlp(block.norm2(x))
    
    return attention_maps


def visualize_attention(
    model,
    image_tensor,
    label,
    img_size=224,
    patch_size=16,
    save_path=None
):
    """
    Visualize attention maps for a single image across all transformer blocks.
    
    Args:
        model: Vision Transformer model
        image_tensor: Preprocessed image tensor
        label: True label index
        img_size: Image size
        patch_size: Patch size
        save_path: Path to save visualization
    """
    device = next(model.parameters()).device
    
    # Get attention maps
    image_batch = image_tensor.unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_batch.to(device))
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    # Denormalize image for display
    img_display = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
    std = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    img_display = img_display * std + mean
    img_display = np.clip(img_display, 0, 1)
    
    # Get attention maps from model
    attention_maps = get_attention_maps(model, image_batch, device)
    
    # Calculate number of patches
    num_patches = (img_size // patch_size) ** 2
    
    # Visualize attention from CLS token to all patches for selected layers
    selected_layers = [0, len(attention_maps)//2, len(attention_maps)-1]  # First, middle, last
    
    fig, axes = plt.subplots(2, len(selected_layers) + 1, figsize=(15, 8))
    
    # Show original image
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title(f'Original Image\nTrue: {CIFAR100_CLASSES[label]}', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(img_display)
    pred_color = 'green' if pred_idx == label else 'red'
    axes[1, 0].set_title(f'Prediction\n{CIFAR100_CLASSES[pred_idx]} ({confidence*100:.1f}%)', 
                         fontsize=10, color=pred_color)
    axes[1, 0].axis('off')
    
    # Visualize attention maps from selected layers
    for idx, layer_idx in enumerate(selected_layers, start=1):
        attn = attention_maps[layer_idx][0]  # (num_heads, num_patches+1, num_patches+1)
        
        # Average over all attention heads
        attn_mean = attn.mean(dim=0)  # (num_patches+1, num_patches+1)
        
        # Get attention from CLS token (first token) to all patches
        cls_attn = attn_mean[0, 1:]  # Exclude attention to CLS itself
        
        # Reshape to 2D grid
        grid_size = int(np.sqrt(num_patches))
        cls_attn_map = cls_attn.reshape(grid_size, grid_size).numpy()
        
        # Upsample to image size for better visualization
        cls_attn_upsampled = np.kron(cls_attn_map, np.ones((patch_size, patch_size)))
        
        # Row 1: Attention heatmap
        im1 = axes[0, idx].imshow(cls_attn_upsampled, cmap='hot', interpolation='bilinear')
        axes[0, idx].set_title(f'Layer {layer_idx+1}\nAttention Heatmap', fontsize=10)
        axes[0, idx].axis('off')
        plt.colorbar(im1, ax=axes[0, idx], fraction=0.046)
        
        # Row 2: Attention overlay on image
        axes[1, idx].imshow(img_display)
        im2 = axes[1, idx].imshow(cls_attn_upsampled, cmap='hot', alpha=0.6, interpolation='bilinear')
        axes[1, idx].set_title(f'Layer {layer_idx+1}\nAttention Overlay', fontsize=10)
        axes[1, idx].axis('off')
    
    plt.suptitle(f'Vision Transformer Attention Visualization\nModel focuses on different regions across layers', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention visualization saved to: {save_path}")
    
    plt.show()


def visualize_head_attention(
    model,
    image_tensor,
    label,
    layer_idx=11,  # Last layer
    img_size=224,
    patch_size=16,
    save_path=None
):
    """
    Visualize attention from individual heads in a specific layer.
    
    Args:
        model: Vision Transformer model
        image_tensor: Preprocessed image tensor
        label: True label
        layer_idx: Which transformer block to visualize
        img_size: Image size
        patch_size: Patch size
        save_path: Path to save
    """
    device = next(model.parameters()).device
    image_batch = image_tensor.unsqueeze(0)
    
    # Get attention maps
    attention_maps = get_attention_maps(model, image_batch, device)
    
    # Get attention from selected layer
    attn = attention_maps[layer_idx][0]  # (num_heads, num_patches+1, num_patches+1)
    num_heads = attn.shape[0]
    num_patches = (img_size // patch_size) ** 2
    grid_size = int(np.sqrt(num_patches))
    
    # Denormalize image
    img_display = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
    std = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    img_display = img_display * std + mean
    img_display = np.clip(img_display, 0, 1)
    
    # Create subplots
    cols = 4
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    axes = axes.flatten()
    
    # Visualize each attention head
    for head_idx in range(num_heads):
        # Get attention from CLS token
        cls_attn = attn[head_idx, 0, 1:]  # (num_patches,)
        cls_attn_map = cls_attn.reshape(grid_size, grid_size).numpy()
        cls_attn_upsampled = np.kron(cls_attn_map, np.ones((patch_size, patch_size)))
        
        # Overlay on image
        axes[head_idx].imshow(img_display)
        axes[head_idx].imshow(cls_attn_upsampled, cmap='hot', alpha=0.6, interpolation='bilinear')
        axes[head_idx].set_title(f'Head {head_idx + 1}', fontsize=10)
        axes[head_idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Multi-Head Attention - Layer {layer_idx + 1}\nTrue Label: {CIFAR100_CLASSES[label]}', 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Head attention visualization saved to: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # Configuration
    checkpoint_path = 'checkpoints/best_model.pth'
    data_dir = '../cifar-100-python/cifar-100-python'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(checkpoint_path, device=device)
    
    # Load test dataset
    test_dataset = CIFAR100Dataset(
        data_dir,
        train=False,
        transform=get_transforms(224, train=False)
    )
    
    # Visualize attention for random samples
    print("\nGenerating attention visualizations...")
    
    for i in range(3):  # Visualize 3 examples
        idx = np.random.randint(len(test_dataset))
        image, label = test_dataset[idx]
        
        print(f"\nExample {i+1}: {CIFAR100_CLASSES[label]}")
        
        # Visualize layer-wise attention
        visualize_attention(
            model,
            image,
            label,
            save_path=f'attention_layers_example_{i+1}.png'
        )
        
        # Visualize individual heads
        visualize_head_attention(
            model,
            image,
            label,
            layer_idx=11,  # Last layer
            save_path=f'attention_heads_example_{i+1}.png'
        )
    
    print("\n[OK] Attention visualizations completed!")

