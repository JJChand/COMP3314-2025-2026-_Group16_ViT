from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from train import train_one_epoch, validate
from model import vit_small_patch16_224

IMG_SIZE = 224

CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

MODEL_MEAN = CIFAR_MEAN
MODEL_STD  = CIFAR_STD

class OxfordPets(Dataset):
    """
    annotations/list.txt: name class_id species breed_id
    splits: annotations/trainval.txt, annotations/test.txt
    Labels in [0, 36].
    """
    def __init__(self, root: Path, split: str, transform=None):
        assert split in {"trainval", "test"}
        self.root = root
        self.transform = transform

        ann_dir = root / "annotations"
        img_dir = root / "images"

        # Parse list.txt -> {basename: (label0, species, breed_id)}
        self.meta: dict[str, tuple[int, int, int]] = {}
        with open(ann_dir / "list.txt", "r") as f:
            rows = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        for ln in rows:
            parts = ln.split()
            if len(parts) < 4:
                continue
            base = parts[0]                   # usually without extension
            cls_id = int(parts[1]) - 1        # 0-based
            species = int(parts[2])
            breed_id = int(parts[3])
            self.meta[base] = (cls_id, species, breed_id)

        with open(ann_dir / f"{split}.txt", "r") as f:
            split_names = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            split_names = [name.split(" ")[0] for name in split_names]

        self.samples: list[tuple[Path, int]] = []
        for base in split_names:
            stem = base[:-4] if base.lower().endswith(".jpg") else base
            img_path = (img_dir / f"{stem}.jpg")
            if stem not in self.meta:
                raise FileNotFoundError(f"{stem} not present in list.txt metadata.")
            label, _, _ = self.meta[stem]
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image file: {img_path}")
            self.samples.append((img_path, label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Better 224x224 augmentations for ViT
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MODEL_MEAN, MODEL_STD),
])

test_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MODEL_MEAN, MODEL_STD),
])

BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 100
BASE_LR = 1e-3
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 37

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    model = vit_small_patch16_224(num_classes=100)
    ckpt = torch.load('checkpoints/best_model_new.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, NUM_CLASSES)
    return model

def freeze_head_only(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True
    elif hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        raise RuntimeError("Could not find classifier head to unfreeze.")

def main():
    root = Path("oxford-pets")  # expects images/ and annotations/ under this
    train_set = OxfordPets(root, split="trainval", transform=train_tf)
    test_set  = OxfordPets(root, split="test",     transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model().to(DEVICE)

    freeze_head_only(model)

    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    save_path = Path("checkpoints/oxford_pets_finetuned.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        va_loss, va_acc = validate(model,     test_loader,  criterion, DEVICE, epoch)
        scheduler.step()

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train: loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
              f"val: loss {va_loss:.4f} acc {va_acc:.2f}%  "
              f"(best {best_acc:.2f}%)")

    print(f"Best checkpoint saved to: {save_path}")

if __name__ == "__main__":
    main()