# Vision Transformer (ViT) Implementation
## COMP3314 2025-2026 Group 16

Reproduction and implementation of **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))

---

## 📋 Project Overview

This project implements a Vision Transformer (ViT) model from scratch and demonstrates:
- **Training from scratch** on CIFAR-100 (100 classes)
- **Attention visualization** to understand model behavior
- **Transfer learning** from CIFAR-100 to CIFAR-10

### Key Results

| Model | Dataset | Method | Accuracy | Top-5 Acc |
|-------|---------|--------|----------|-----------|
| ViT-Small | CIFAR-100 | From Scratch | **66.65%** | - |
| ViT-Small | CIFAR-10 | Transfer (Frozen Head) | 12.21% | 53.53% |
| ViT-Small | CIFAR-10 | Transfer (Last 4 Blocks) | **63.66%** | **96.75%** |

---

## 🗂️ Project Structure

```
COMP3314-2025-2026-_Group16_ViT/
├── src/
│   ├── model.py                    # ViT architecture implementation
│   ├── train.py                    # Training script for CIFAR-100
│   ├── utils.py                    # Evaluation and utilities
│   ├── finetune_cifar10.py        # Transfer learning script
│   ├── visualization.py            # Training progress visualization
│   ├── visualize_attention.py     # Attention map visualization
│   ├── checkpoints/               # CIFAR-100 trained models
│   │   ├── best_model.pth
│   │   └── training_history.json
│   └── cifar10_finetuned_optionA/ # CIFAR-10 fine-tuned models
│       ├── best_model.pth
│       └── training_history.json
├── cifar-100-python/              # CIFAR-100 dataset
├── cifar-10-python/               # CIFAR-10 dataset
├── demo.ipynb                     # 🌟 Interactive demo notebook
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n vit python=3.8
conda activate vit

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# CIFAR-100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz

# CIFAR-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
```

### 3. Run the Demo Notebook

**Best way to explore the project:**

```bash
jupyter notebook demo.ipynb
```

The notebook includes:
- ✅ Training visualization for CIFAR-100
- ✅ Attention map visualization
- ✅ Transfer learning to CIFAR-10
- ✅ Complete results and analysis

---

## 📚 Detailed Usage

### Training on CIFAR-100 (From Scratch)

```bash
cd src
python train.py --model small --epochs 300 --lr 1e-3 --batch-size 128
```

**Key hyperparameters:**
- Model: ViT-Small (12 layers, 384 dim, 6 heads)
- Optimizer: AdamW (lr=1e-3, weight_decay=0.1)
- Scheduler: Warmup (10 epochs) + Cosine Annealing
- Data augmentation: RandAugment, RandomErasing
- Gradient clipping: 1.0
- Total epochs: 300

### Visualizing Attention Maps

```bash
cd src
python visualize_attention.py
```

Generates:
- `attention_layers_example_*.png` - Layer-wise attention progression
- `attention_heads_example_*.png` - Multi-head attention patterns

### Transfer Learning (CIFAR-100 → CIFAR-10)

```bash
cd src
python finetune_cifar10.py
```

**Strategy (Option A):**
- Freeze: Patch embedding + first 8 transformer blocks
- Unfreeze: Last 4 transformer blocks + classification head
- Learning rate: 1e-4 (lower than from-scratch training)
- Epochs: 50 (fewer than from-scratch)

**Critical**: Uses CIFAR-100 normalization statistics because the backbone was trained on CIFAR-100!

### Model Evaluation

```bash
cd src
python utils.py  # Run evaluation functions
```

---

## 🏗️ Model Architecture

### ViT-Small Configuration

```python
{
    'image_size': 224,
    'patch_size': 16,
    'num_classes': 100,  # or 10 for CIFAR-10
    'embed_dim': 384,
    'depth': 12,
    'num_heads': 6,
    'mlp_ratio': 4,
    'dropout': 0.1,
    'attn_dropout': 0.1
}
```

**Total parameters**: ~21.6M

### Architecture Components

1. **Patch Embedding**: 16×16 patches → 384-dim embeddings
2. **Position Embedding**: Learnable 1D position encodings
3. **Transformer Encoder**: 12 layers of:
   - Multi-Head Self-Attention (6 heads)
   - MLP (384 → 1536 → 384)
   - LayerNorm, Residual connections
4. **Classification Head**: Linear layer (384 → num_classes)

---

## 📊 Training Details

### CIFAR-100 Training

**Data Augmentation:**
- RandomCrop(32, padding=4)
- RandomHorizontalFlip(p=0.5)
- RandAugment(num_ops=2, magnitude=9)
- RandomErasing(p=0.25)

**Learning Rate Schedule:**
```
Warmup: 0 → 1e-3 (10 epochs, linear)
Cosine: 1e-3 → 1e-6 (290 epochs, cosine decay)
```

**Results:**
- Best validation accuracy: 66.65% (Epoch 266)
- Training time: ~24 hours (RTX GPU)
- Convergence: Stable after 200 epochs

### CIFAR-10 Transfer Learning

**Option A Results (Unfreeze Last 4 Blocks):**
- Best validation accuracy: 63.66% (Epoch 42)
- Top-5 accuracy: 96.75%
- Training time: ~2 hours (RTX GPU)

**Per-Class Accuracy:**
```
Automobile:  73.3%  ⭐ Best
Ship:        72.9%
Frog:        72.5%
Truck:       69.7%
Airplane:    68.9%
Horse:       67.4%
Dog:         65.2%
Deer:        56.2%
Bird:        56.0%
Cat:         34.5%  ⚠️ Most challenging
```

---

## 🔬 Key Insights

### 1. Attention Patterns
- **Early layers**: Broad, global attention
- **Middle layers**: Feature aggregation and grouping
- **Late layers**: Focused on discriminative regions

### 2. Transfer Learning Strategy
- **Failed**: Freezing entire backbone (12.21% accuracy)
- **Success**: Unfreezing last 4 blocks (63.66% accuracy)
- **Lesson**: Deep features need adaptation for new tasks

### 3. Implementation Challenges
- **Normalization**: Must use source dataset statistics for frozen backbone
- **Learning rate**: Fine-tuning requires lower LR than from-scratch
- **Data augmentation**: RandAugment requires uint8 images
- **GPU compatibility**: Modern GPUs (RTX 50-series) need latest PyTorch

---

## 📈 Visualizations

The project generates several visualizations:

1. **Training Progress** (`visualization.py`)
   - Loss curves (train/val)
   - Accuracy curves (train/val)
   - Learning rate schedule
   - Overfitting analysis

2. **Attention Maps** (`visualize_attention.py`)
   - Layer-wise attention evolution
   - Per-head attention patterns
   - Prediction overlays

3. **Evaluation Results** (`utils.py`)
   - Confusion matrix
   - Per-class accuracy
   - Top-5 accuracy analysis
   - Best/worst performing classes

---

## 🛠️ Technical Requirements

### Hardware
- GPU: NVIDIA GPU with CUDA support (>= 8GB VRAM recommended)
- RAM: >= 16GB
- Storage: ~2GB for datasets + models

### Software
```txt
Python >= 3.8
PyTorch >= 2.0
torchvision
numpy
matplotlib
seaborn
tqdm
Pillow
```

See `requirements.txt` for exact versions.

---

## 📖 References

### Paper
- Dosovitskiy, A., et al. (2020). **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**. 
  *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

### Dataset
- Krizhevsky, A. (2009). **Learning Multiple Layers of Features from Tiny Images**. 
  [Technical Report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

### Implementation Guidance
- Official ViT implementation: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)
- PyTorch timm library: [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

---

## 👥 Team

**COMP3314 2025-2026 Group 16**

---

## 📝 License

This project is for educational purposes as part of COMP3314 coursework.

---

## 🙏 Acknowledgments

- Original ViT paper authors
- CIFAR dataset creators
- PyTorch and timm library contributors
- Course instructors and TAs

---

## 🔗 Quick Links

- [Paper PDF](./2010.11929v2.pdf)
- [Interactive Demo](./demo.ipynb) ⭐ **Start here!**
- [Training Script](./src/train.py)
- [Model Architecture](./src/model.py)
- [Transfer Learning](./src/finetune_cifar10.py)

---

**Last Updated**: November 2025
