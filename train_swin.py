import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from dataset import KaggleImageDataset, resolve_image_dir


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin V2-S with EMA for meteorite classification.")
    parser.add_argument("--data_dir", type=str, default="/data/final")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--labels_csv", type=str, default="train_labels.csv")
    parser.add_argument("--train_image_folder", type=str, default="train_images")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device, but CUDA is not available.")
    return device


def create_run_dir(output_root, run_name):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = f"swin_v2_s_ema_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.RandomEqualize(p=0.7),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.05),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(num_classes, pretrained=True):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.swin_v2_s(weights=weights)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def make_loaders(args, device):
    data_dir = Path(args.data_dir)
    labels_path = data_dir / args.labels_csv
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    df = pd.read_csv(labels_path)
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError("Labels CSV must contain 'id' and 'label' columns.")

    encoder = LabelEncoder()
    targets = encoder.fit_transform(df["label"])
    train_df, val_df, train_targets, val_targets = train_test_split(
        df,
        targets,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=targets,
    )

    train_transform, eval_transform = get_transforms(args.image_size)
    train_image_dir = resolve_image_dir(data_dir, args.train_image_folder)
    train_dataset = KaggleImageDataset(
        train_df.reset_index(drop=True),
        train_image_dir,
        transform=train_transform,
        labels=train_targets,
    )
    val_dataset = KaggleImageDataset(
        val_df.reset_index(drop=True),
        train_image_dir,
        transform=eval_transform,
        labels=val_targets,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    classes = [item.item() if hasattr(item, "item") else item for item in encoder.classes_]
    return train_loader, val_loader, classes


def train_one_epoch(model, loader, criterion, optimizer, ema, device, use_amp):
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = criterion(model(images), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, ema, device, use_amp):
    model.eval()
    ema.apply_shadow()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    ema.restore()
    return total_loss / len(loader.dataset), f1_score(all_labels, all_preds, average="macro")


def save_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    run_dir = create_run_dir(args.output_root, args.run_name)

    train_loader, val_loader, classes = make_loaders(args, device)
    model = build_model(num_classes=len(classes), pretrained=True).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_path = run_dir / "best_model.pth"
    metrics_path = run_dir / "metrics.csv"
    save_json(
        run_dir / "args.json",
        {
            **vars(args),
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_path),
            "model_name": "swin_v2_s",
            "classes": classes,
        },
    )

    best_f1 = -1.0
    bad_epochs = 0
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_f1", "is_best"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, ema, device, use_amp)
            val_loss, val_f1 = evaluate(model, val_loader, criterion, ema, device, use_amp)
            scheduler.step()
            is_best = val_f1 > best_f1

            if is_best:
                best_f1 = val_f1
                bad_epochs = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema_shadow": {key: value.detach().cpu() for key, value in ema.shadow.items()},
                        "model_name": "swin_v2_s",
                        "num_classes": len(classes),
                        "classes": classes,
                        "image_size": args.image_size,
                        "val_f1": val_f1,
                        "args": vars(args),
                    },
                    best_path,
                )
            else:
                bad_epochs += 1

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_f1": f"{val_f1:.6f}",
                    "is_best": int(is_best),
                }
            )
            handle.flush()
            print(
                f"[{epoch}/{args.epochs}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} best_f1={best_f1:.4f}"
            )

            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Best Swin checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
