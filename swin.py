"""Method 19: Swin V2-S + EMA (Exponential Moving Average).
Same as M12 but with EMA shadow weights for inference.
Expected: Test F1 > 0.6666 (M12 baseline)."""

import os, time, random, copy
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

DATA_ROOT = "/data/meteorite-identification-stage-2"
OUTPUT_DIR = "/data/Swin"
seed = 42
IMG_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE = 10
EMA_DECAY = 0.999

random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Dataset ──────────────────────────────────────────────────────────

class StoneDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root = root; self.split = split; self.transforms = transforms
        if split == "train":
            df = pd.read_csv(os.path.join(root, "train_labels.csv"))
            self.ids = df["id"].tolist()
            self.labels = df["label"].astype(int).tolist()
            self.image_paths = [self._resolve(os.path.join(root, "train_images"), i) for i in self.ids]
        else:
            df = pd.read_csv(os.path.join(root, "sample_submission.csv"))
            self.ids = df["id"].tolist(); self.labels = None
            self.image_paths = [self._resolve(os.path.join(root, "test_images"), i) for i in self.ids]

    def _resolve(self, folder, image_id):
        p = os.path.join(folder, image_id)
        if os.path.exists(p): return p
        for ext in [".jpg",".png",".jpeg",".JPG",".PNG",".JPEG"]:
            p2 = os.path.join(folder, image_id + ext)
            if os.path.exists(p2): return p2
        matches = list(Path(folder).rglob("*"))
        for m in matches:
            if m.is_file() and (m.name == image_id or m.stem == Path(image_id).stem):
                return str(m)
        raise FileNotFoundError(image_id)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transforms: img = self.transforms(img)
        if self.labels is not None: return img, self.labels[idx]
        return img, self.image_paths[idx]

# ── EMA ──────────────────────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model; self.decay = decay
        self.shadow = {}; self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()

# ── Augmentation (IDDENTICAL to M12) ─────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.3, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.RandomEqualize(p=0.7),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.05),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

print(f"M19 EMA: {IMG_SIZE}px, anti-brightness + multi-scale (same as M12) + EMA(decay={EMA_DECAY})")

# ── Model ────────────────────────────────────────────────────────────

model = models.swin_v2_s(weights="IMAGENET1K_V1").to(device)
model.head = nn.Linear(model.head.in_features, 2).to(device)
ema = ModelEMA(model, decay=EMA_DECAY)
n_params = sum(p.numel() for p in model.parameters())
print(f"Swin V2-S: {n_params/1e6:.1f}M params + EMA")

# ── Data loading ─────────────────────────────────────────────────────

train_full = StoneDataset(DATA_ROOT, "train", train_transform)
val_full = StoneDataset(DATA_ROOT, "train", val_test_transform)
test_ds = StoneDataset(DATA_ROOT, "test", val_test_transform)

all_idx = np.arange(len(train_full))
all_labels = np.array(train_full.labels)
idx0 = list(all_idx[all_labels == 0]); idx1 = list(all_idx[all_labels == 1])
random.Random(seed).shuffle(idx0); random.Random(seed).shuffle(idx1)
val_idx = idx0[:250] + idx1[:250]
train_idx = list(set(all_idx) - set(val_idx))

train_ds = Subset(train_full, train_idx); val_ds = Subset(val_full, val_idx)
trainloader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
valloader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
testloader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ── Optimizer ────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training functions ───────────────────────────────────────────────

def train_epoch():
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(trainloader, desc="train", leave=False):
        images = images.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward(); optimizer.step()
        ema.update()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(train_ds)

@torch.no_grad()
def eval_epoch(loader, use_ema=True):
    model.eval()
    if use_ema: ema.apply_shadow()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images = images.to(device); labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    if use_ema: ema.restore()
    return total_loss / len(loader.dataset), f1_score(all_labels, all_preds, average="macro")

# ── Training loop ────────────────────────────────────────────────────

best_val_f1, best_path = -1.0, f"{OUTPUT_DIR}/method19_swin_ema_best.pth"
bad_epochs, t0 = 0, time.time()

for epoch in range(EPOCHS):
    tr_loss = train_epoch()
    va_loss, va_f1 = eval_epoch(valloader, use_ema=True)
    scheduler.step()

    if va_f1 > best_val_f1:
        best_val_f1 = va_f1; bad_epochs = 0
        torch.save({
            "model": model.state_dict(),
            "ema_shadow": {k: v.clone() for k, v in ema.shadow.items()},
            "val_f1": va_f1,
        }, best_path)
    else:
        bad_epochs += 1

    print(f"[{epoch+1}/{EPOCHS}] TrL:{tr_loss:.4f} | VaF1(EMA):{va_f1:.4f} | Best:{best_val_f1:.4f} | Bad:{bad_epochs}/{PATIENCE}")
    if bad_epochs >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\nDone in {(time.time()-t0)/60:.1f} min | Best Val F1 (EMA): {best_val_f1:.4f}")

# ── Predict function (general, returns per-sample results) ────────────

@torch.no_grad()
def predict_with_confidence(model, loader, device):
    """Returns dicts mapping image_basename -> prediction/confidence/probabilities."""
    model.eval()
    id_to_pred, id_to_conf = {}, {}
    id_to_prob0, id_to_prob1 = {}, {}
    for images, paths in tqdm(loader, desc="predict", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1).cpu().numpy().tolist()
        confs = probs.max(1).values.cpu().numpy().tolist()
        p0 = probs[:, 0].cpu().numpy().tolist()
        p1 = probs[:, 1].cpu().numpy().tolist()
        for pred, c, prob0, prob1, path in zip(preds, confs, p0, p1, paths):
            name = os.path.basename(path)
            id_to_pred[name] = int(pred)
            id_to_conf[name] = float(c)
            id_to_prob0[name] = float(prob0)
            id_to_prob1[name] = float(prob1)
    return id_to_pred, id_to_conf, id_to_prob0, id_to_prob1


# ── Load best checkpoint ─────────────────────────────────────────────

ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model"])

# Raw weights
va_labels, va_preds_raw = [], []
for images, labels in valloader:
    images = images.to(device)
    va_preds_raw.extend(model(images).argmax(1).cpu().numpy().tolist())
    va_labels.extend(labels.numpy().tolist())
raw_f1 = f1_score(va_labels, va_preds_raw, average="macro")

# ── Apply EMA weights ────────────────────────────────────────────────

for name, param in model.named_parameters():
    if name in ckpt["ema_shadow"]:
        param.data.copy_(ckpt["ema_shadow"][name])

# ── Validation set: detailed predictions with confidence ─────────────

va_labels, va_preds_ema, va_confs, va_prob0, va_prob1 = [], [], [], [], []
with torch.no_grad():
    for images, labels in valloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        va_preds_ema.extend(probs.argmax(1).cpu().numpy().tolist())
        va_confs.extend(probs.max(1).values.cpu().numpy().tolist())
        va_prob0.extend(probs[:, 0].cpu().numpy().tolist())
        va_prob1.extend(probs[:, 1].cpu().numpy().tolist())
        va_labels.extend(labels.numpy().tolist())

ema_f1 = f1_score(va_labels, va_preds_ema, average="macro")

# Build validation DataFrame with image ids
val_img_ids = [os.path.basename(val_full.image_paths[i]) for i in val_idx]
val_df = pd.DataFrame({
    "id": val_img_ids,
    "true_label": va_labels,
    "pred_label": va_preds_ema,
    "confidence": va_confs,
    "prob_class0": va_prob0,
    "prob_class1": va_prob1,
})
val_csv = f"{OUTPUT_DIR}/method19_swin_ema_val.csv"
val_df.to_csv(val_csv, index=False)

print(f"\nRaw Val F1:  {raw_f1:.4f}")
print(f"EMA Val F1:   {ema_f1:.4f}")
print(f"EMA Delta:    {ema_f1-raw_f1:+.4f}")
print(f"Confusion (EMA):\n{confusion_matrix(va_labels, va_preds_ema)}")
print(f"Val confidence: mean={np.mean(va_confs):.4f}, median={np.median(va_confs):.4f}, min={np.min(va_confs):.4f}")
print(f"Val results saved: {val_csv}")

# ── Test set: predictions with confidence ────────────────────────────

test_preds, test_confs, test_prob0, test_prob1 = predict_with_confidence(model, testloader, device)

template = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sub = template.copy()
sub["label"] = sub["id"].map(test_preds).astype(int)
sub["confidence"] = sub["id"].map(test_confs)

output_csv = f"{OUTPUT_DIR}/method19_swin_ema_submission.csv"
sub.to_csv(output_csv, index=False)

# Save detailed test probabilities
test_detail_df = pd.DataFrame({
    "id": sub["id"],
    "label": sub["label"],
    "confidence": sub["confidence"],
    "prob_class0": sub["id"].map(test_prob0),
    "prob_class1": sub["id"].map(test_prob1),
})
detail_csv = f"{OUTPUT_DIR}/method19_swin_ema_test_probs.csv"
test_detail_df.to_csv(detail_csv, index=False)

print(f"\nTest distribution: {sub['label'].value_counts().to_dict()}")
print(f"Test confidence: mean={sub['confidence'].mean():.4f}, median={sub['confidence'].median():.4f}, min={sub['confidence'].min():.4f}")
print(f"Summary: best_val_f1(EMA)={best_val_f1:.4f} | EMA Delta on Val={ema_f1-raw_f1:+.4f}")
print(f"Test submission saved: {output_csv}")
print(f"Test probabilities saved: {detail_csv}")
print(f"\n>>> Compare with M12(Test F1=0.6666). Hypothesized: EMA > 0.6666")
