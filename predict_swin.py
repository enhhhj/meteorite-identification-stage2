import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from dataset import KaggleImageDataset
from inference_utils import (
    load_checkpoint,
    make_test_df,
    positive_class_index,
    resolve_device,
    save_probability_csv,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Swin V2-S EMA probabilities.")
    parser.add_argument("--data_dir", type=str, default="/data/final")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="weights/swin_ema_best.pth")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", type=str, default="outputs/swin_probs.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    return parser.parse_args()


def get_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(num_classes):
    model = models.swin_v2_s(weights=None)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def apply_ema_shadow(model, checkpoint):
    ema_shadow = checkpoint.get("ema_shadow") if isinstance(checkpoint, dict) else None
    if not ema_shadow:
        return

    for name, param in model.named_parameters():
        if name in ema_shadow:
            param.data.copy_(ema_shadow[name].to(param.device))


@torch.no_grad()
def predict_probabilities(model, loader, positive_index, device, use_amp, tta):
    model.eval()
    all_ids = []
    all_probs = []

    for images, image_ids in tqdm(loader, desc="predict", leave=False):
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            probs = torch.softmax(model(images), dim=1)
            if tta == "hflip":
                flipped = torch.flip(images, dims=[3])
                probs = (probs + torch.softmax(model(flipped), dim=1)) / 2.0

        all_ids.extend([str(image_id) for image_id in image_ids])
        all_probs.extend(probs[:, positive_index].cpu().numpy().tolist())

    return all_ids, [float(prob) for prob in all_probs]


def main():
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = device.type == "cuda"

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    classes = checkpoint.get("classes", [0, 1]) if isinstance(checkpoint, dict) else [0, 1]
    num_classes = int(checkpoint.get("num_classes", len(classes))) if isinstance(checkpoint, dict) else len(classes)
    image_size = int(checkpoint.get("image_size", args.image_size)) if isinstance(checkpoint, dict) else args.image_size
    positive_index = positive_class_index(classes, num_classes)

    test_df, test_image_dir = make_test_df(args.data_dir, args.test_dir)
    dataset = KaggleImageDataset(test_df, test_image_dir, transform=get_transform(image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint_state_dict(checkpoint), strict=True)
    model.to(device)
    apply_ema_shadow(model, checkpoint)

    image_ids, probabilities = predict_probabilities(
        model=model,
        loader=loader,
        positive_index=positive_index,
        device=device,
        use_amp=use_amp,
        tta=args.tta,
    )
    output_path = save_probability_csv(image_ids, probabilities, args.out, threshold=args.threshold)
    print(f"Saved Swin probabilities: {output_path}")


if __name__ == "__main__":
    main()
