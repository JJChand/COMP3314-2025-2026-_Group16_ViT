from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from train import train_one_epoch, validate
from model import vit_small_patch16_224

IMG_SIZE = 224
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

root = "cifar-10"
train_set = datasets.CIFAR10(root, train=True,  download=True, transform=train_tf)
test_set  = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

BATCH_SIZE = 128
NUM_WORKERS = 4

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
trained_model = vit_small_patch16_224(num_classes=100)
checkpoint = torch.load('checkpoints/best_model_new.pth', map_location=DEVICE, weights_only=False)
trained_model.load_state_dict(checkpoint['model_state_dict'])
in_features = trained_model.head.in_features
trained_model.head = nn.Linear(in_features, 10)
trained_model.to(DEVICE)

for name, p in trained_model.named_parameters():
    p.requires_grad = False
    for p in trained_model.parameters():
        if p.ndim == 2 and p.shape[-1] == 10:  # crude way to catch the last Linear
            p.requires_grad = True
    # Better: explicitly unfreeze just model.fc / model.head / model.classifier
    if hasattr(trained_model, "fc"):
        for p in trained_model.fc.parameters():
            p.requires_grad = True
    if hasattr(trained_model, "head"):
        for p in trained_model.head.parameters(): p.requires_grad = True
    if hasattr(trained_model, "classifier"):
        for p in trained_model.classifier.parameters(): p.requires_grad = True

EPOCHS = 100
BASE_LR = 1e-3
WEIGHT_DECAY = 5e-4

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Only pass trainable params to the optimizer
params = [p for p in trained_model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
save_path = Path("checkpoints/cifar10_finetuned.pth")

if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(trained_model, train_loader, criterion, optimizer, DEVICE, epoch)
        va_loss, va_acc = validate(trained_model, test_loader, criterion, DEVICE, epoch)
        scheduler.step()

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(trained_model.state_dict(), save_path)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train: loss {tr_loss:.4f} acc {tr_acc :.2f}% | "
              f"val: loss {va_loss:.4f} acc {va_acc :.2f}%  "
              f"(best {best_acc :.2f}%)")

    print(f"Best checkpoint saved to: {save_path}")