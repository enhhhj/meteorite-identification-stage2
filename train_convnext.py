import argparse
import csv
import json
import logging
import re
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
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from tqdm import tqdm

from dataset import KaggleImageDataset, resolve_image_dir


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny for Kaggle image classification.")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="convnext_tiny", choices=["convnext_tiny"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--val_size", type=float, default=0.2)
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be >= 0")
    if args.image_size < 32:
        raise ValueError("--image_size must be >= 32")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")
    if args.min_delta < 0:
        raise ValueError("--min_delta must be >= 0")
    if not 0 < args.val_size < 1:
        raise ValueError("--val_size must be between 0 and 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def compact_float(value: float) -> str:
    if value == 0:
        return "0"
    text = f"{value:.0e}" if abs(value) < 1e-3 or abs(value) >= 1e3 else f"{value:g}"
    return text.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._-") or "run"


def build_run_name(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        return safe_name(args.run_name)
    return safe_name(
        f"{args.model_name}_ep{args.epochs}_bs{args.batch_size}_lr{compact_float(args.lr)}_"
        f"wd{compact_float(args.weight_decay)}_img{args.image_size}_seed{args.seed}_{timestamp}"
    )


def create_run_dir(output_root: str, run_name: str) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_dir = output_root / run_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return run_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback = output_root / f"{run_name}_{timestamp}"
    index = 1
    while fallback.exists():
        fallback = output_root / f"{run_name}_{timestamp}_{index:02d}"
        index += 1

    fallback.mkdir(parents=True)
    return fallback


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("convnext_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device, but CUDA is not available.")

    return device


def json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_safe)


def get_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.08,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(
                p=0.15,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name != "convnext_tiny":
        raise ValueError(f"Unsupported model_name: {model_name}")

    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


def image_average_hash(path: Path, hash_size: int = 8) -> str:
    """Compute simple average hash for duplicate / near-duplicate grouping."""
    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size, hash_size))
        arr = np.asarray(img, dtype=np.float32)

    avg = arr.mean()
    bits = arr > avg
    return "".join("1" if x else "0" for x in bits.flatten())


def add_image_groups(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Add image_group column. Duplicate images share the same group."""
    df = df.copy()
    groups = []

    for image_id in df["id"].astype(str):
        image_path = image_dir / image_id

        if not image_path.exists():
            stem = Path(image_id).stem
            found = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                candidate = image_dir / f"{stem}{ext}"
                if candidate.exists():
                    found = candidate
                    break
            image_path = found

        if image_path is None or not Path(image_path).exists():
            groups.append(f"missing_{image_id}")
        else:
            try:
                groups.append(image_average_hash(Path(image_path)))
            except Exception:
                groups.append(f"broken_{image_id}")

    df["image_group"] = groups
    return df


def group_stratified_split(df: pd.DataFrame, targets: np.ndarray, val_size: float, seed: int):
    """Split train/val while keeping duplicate image groups in the same split."""
    n_splits = max(2, round(1 / val_size))

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    groups = df["image_group"].values

    train_idx, val_idx = next(splitter.split(df, targets, groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_targets = targets[train_idx]
    val_targets = targets[val_idx]

    overlap = set(train_df["image_group"]) & set(val_df["image_group"])
    if overlap:
        raise RuntimeError(f"Duplicate image groups leaked into both train and val: {len(overlap)}")

    return train_df, val_df, train_targets, val_targets

def class_weights_from_targets(targets: np.ndarray, num_classes: int):
    counts = np.bincount(targets, minlength=num_classes)

    if counts.min() == 0 or counts.max() == counts.min():
        return None

    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(args, train_df, val_df, train_targets, val_targets, device):
    train_transform, val_transform = get_transforms(args.image_size)
    train_image_dir = resolve_image_dir(args.data_dir, "train_images")

    train_dataset = KaggleImageDataset(
        train_df,
        train_image_dir,
        label_col="label",
        transform=train_transform,
        labels=train_targets,
    )

    val_dataset = KaggleImageDataset(
        val_df,
        train_image_dir,
        label_col="label",
        transform=val_transform,
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

    return train_loader, val_loader, train_image_dir


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    val_loss = running_loss / len(loader.dataset)
    val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return val_loss, val_f1


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, mode: str = "max"):
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float):
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return True, False

        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.bad_epochs = 0
            return True, False

        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


def make_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    classes,
    epoch,
    train_loss,
    val_loss,
    val_f1,
    best_val_loss,
    best_val_f1,
):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_name": args.model_name,
        "num_classes": len(classes),
        "classes": classes,
        "image_size": args.image_size,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "monitor": "val_f1",
        "args": vars(args),
    }


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    run_name = build_run_name(args)
    run_dir = create_run_dir(args.output_root, run_name)
    args.run_name = run_dir.name

    log_path = run_dir / "train.log"
    logger = setup_logger(log_path)

    best_path = run_dir / "best_model.pth"
    last_path = run_dir / "last_model.pth"
    args_path = run_dir / "args.json"
    metrics_path = run_dir / "metrics.csv"
    submission_path = run_dir / "submission.csv"

    save_json(
        args_path,
        {
            **vars(args),
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
            "submission_path": str(submission_path),
            "monitor": "val_f1",
        },
    )

    logger.info("Run directory: %s", run_dir)
    logger.info("Args saved to: %s", args_path)

    data_dir = Path(args.data_dir)

    labels_path = data_dir / "train_labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    sample_path = data_dir / "sample_submission.csv"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing sample submission file for final prediction: {sample_path}")

    df = pd.read_csv(labels_path)

    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError("train_labels.csv must contain 'id' and 'label' columns")

    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(df["label"])

    classes = [
        value.item() if hasattr(value, "item") else value
        for value in label_encoder.classes_
    ]

    num_classes = len(classes)
    class_counts = np.bincount(targets, minlength=num_classes)

    stratify = targets if num_classes > 1 and class_counts.min() >= 2 else None

    train_df, val_df, train_targets, val_targets = train_test_split(
        df,
        targets,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify,
    )

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"

    train_loader, val_loader, train_image_dir = make_loaders(
        args,
        train_df,
        val_df,
        train_targets,
        val_targets,
        device,
    )

    logger.info("Data directory: %s", data_dir)
    logger.info("Train image directory: %s", train_image_dir)
    logger.info("Train samples: %d | Val samples: %d", len(train_df), len(val_df))
    logger.info("Classes: %s", classes)
    logger.info("Class counts: %s", class_counts.tolist())
    logger.info("Device: %s | AMP: %s", device, use_amp)
    logger.info("Monitor metric: val_f1")

    model = build_model(
        args.model_name,
        num_classes=num_classes,
        pretrained=True,
    ).to(device)

    weights = class_weights_from_targets(train_targets, num_classes)

    if weights is not None:
        logger.info("Using class weights: %s", weights.tolist())
        weights = weights.to(device)
    else:
        logger.info("Class weights disabled: classes are balanced or unavailable.")

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.05,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode="max",
    )

    logger.info("Best checkpoint: %s", best_path)
    logger.info("Last checkpoint: %s", last_path)
    logger.info("Metrics CSV: %s", metrics_path)

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_f1",
                "lr",
                "is_best",
                "bad_epochs",
            ],
        )

        writer.writeheader()

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        stopped_early = False

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                use_amp,
            )

            val_loss, val_f1 = validate(
                model,
                val_loader,
                criterion,
                device,
                use_amp,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            is_best, should_stop = early_stopping.step(val_f1)

            if is_best:
                best_val_loss = val_loss
                best_val_f1 = val_f1

            checkpoint = make_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                classes=classes,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_f1=val_f1,
                best_val_loss=best_val_loss,
                best_val_f1=best_val_f1,
            )

            torch.save(checkpoint, last_path)

            if is_best:
                torch.save(checkpoint, best_path)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_f1": f"{val_f1:.6f}",
                    "lr": f"{current_lr:.10g}",
                    "is_best": int(is_best),
                    "bad_epochs": early_stopping.bad_epochs,
                }
            )
            f.flush()

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | lr=%.6g | best_f1=%s | bad_epochs=%d/%d",
                epoch,
                args.epochs,
                train_loss,
                val_loss,
                val_f1,
                current_lr,
                is_best,
                early_stopping.bad_epochs,
                args.patience,
            )

            scheduler.step()

            if should_stop:
                stopped_early = True
                logger.info(
                    "Early stopping triggered at epoch %d. Best val_f1=%.6f, best val_loss=%.6f",
                    epoch,
                    best_val_f1,
                    best_val_loss,
                )
                break

    if not best_path.is_file():
        logger.warning("Best checkpoint was not created during training; copying last checkpoint as best.")
        checkpoint = torch.load(last_path, map_location="cpu")
        torch.save(checkpoint, best_path)

    logger.info("Training finished. stopped_early=%s", stopped_early)
    logger.info("Best val_f1=%.6f | best val_loss=%.6f", best_val_f1, best_val_loss)

    from predict_convnext import run_prediction

    logger.info("Creating submission with best F1 checkpoint.")

    run_prediction(
        data_dir=args.data_dir,
        checkpoint=best_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_path=submission_path,
        device_arg=args.device,
        num_workers=args.num_workers,
        logger=logger,
    )

    logger.info("Submission saved to: %s", submission_path)
    logging.shutdown()


if __name__ == "__main__":
    main()