import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny
from tqdm import tqdm

from dataset import KaggleImageDataset, list_image_files, resolve_image_dir
from inference_utils import positive_class_index


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with a trained ConvNeXt-Tiny checkpoint.")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_convnext_tiny.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--out", "--output_path", dest="output_path", type=str, default="outputs/convnext_probs.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device, but CUDA is not available.")

    return device


def load_checkpoint(path: Path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name != "convnext_tiny":
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def get_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def normalize_output_path(output_path) -> Path:
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".csv":
        output_path = output_path / "submission.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def log_message(logger, message, *args):
    if logger is not None:
        logger.info(message, *args)
    else:
        print(message % args if args else message)


def make_test_df(data_dir: Path, test_image_dir: Path, logger=None) -> pd.DataFrame:
    """
    Use current real test_images first.
    If sample_submission.csv exists, keep its order only for ids that actually exist.
    """
    real_test_ids = [p.name for p in list_image_files(test_image_dir)]

    if not real_test_ids:
        raise FileNotFoundError(f"No test images found in {test_image_dir}")

    sample_path = data_dir / "sample_submission.csv"

    if sample_path.is_file():
        sample_df = pd.read_csv(sample_path)

        if "id" not in sample_df.columns:
            raise ValueError("sample_submission.csv must contain 'id' column")

        real_set = set(real_test_ids)
        sample_ids = sample_df["id"].astype(str).tolist()

        ordered_ids = [x for x in sample_ids if x in real_set]
        missing_from_sample = sorted(real_set - set(sample_ids))
        final_ids = ordered_ids + missing_from_sample

        log_message(logger, "Real test image count: %d", len(real_test_ids))
        log_message(logger, "sample_submission row count: %d", len(sample_ids))
        log_message(logger, "Final submission row count: %d", len(final_ids))

        if len(final_ids) != len(real_test_ids):
            raise RuntimeError(
                f"Final ids {len(final_ids)} != real test images {len(real_test_ids)}. "
                "Check duplicate filenames."
            )

        return pd.DataFrame({"id": final_ids})

    log_message(logger, "sample_submission.csv not found. Using sorted test image filenames.")
    log_message(logger, "Real test image count: %d", len(real_test_ids))

    return pd.DataFrame({"id": sorted(real_test_ids)})


def make_tta_batches(images: torch.Tensor):
    """
    images shape: [B, C, H, W]
    TTA variants:
    1. original
    2. horizontal flip
    3. vertical flip
    4. horizontal + vertical flip
    """
    return [
        images,
        torch.flip(images, dims=[3]),
        torch.flip(images, dims=[2]),
        torch.flip(images, dims=[2, 3]),
    ]


@torch.no_grad()
def predict(model, loader, classes, num_classes, device, use_amp, tta=True):
    model.eval()
    id_to_prob = {}
    positive_index = positive_class_index(classes, num_classes)

    for images, image_ids in tqdm(loader, desc="predict", leave=False):
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if tta:
                prob_sum = None
                for tta_images in make_tta_batches(images):
                    outputs = model(tta_images)
                    probs = torch.softmax(outputs, dim=1)
                    prob_sum = probs if prob_sum is None else prob_sum + probs
                avg_probs = prob_sum / 4.0
                positive_probs = avg_probs[:, positive_index].cpu().numpy().tolist()
            else:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                positive_probs = probs[:, positive_index].cpu().numpy().tolist()

        for image_id, prob in zip(image_ids, positive_probs):
            id_to_prob[str(image_id)] = float(prob)

    return id_to_prob


def run_prediction(
    data_dir,
    checkpoint,
    test_dir=None,
    batch_size=64,
    image_size=None,
    output_path="submission.csv",
    device_arg="auto",
    num_workers=2,
    logger=None,
    tta=True,
    threshold=0.5,
):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint)
    output_path = normalize_output_path(output_path)

    device = resolve_device(device_arg)
    use_amp = device.type == "cuda"

    test_image_dir = Path(test_dir) if test_dir else resolve_image_dir(data_dir, "test_images")
    test_df = make_test_df(data_dir, test_image_dir, logger=logger)

    checkpoint_data = load_checkpoint(checkpoint_path, device)

    num_classes = int(checkpoint_data["num_classes"])
    model_name = checkpoint_data.get("model_name", "convnext_tiny")
    classes = checkpoint_data.get("classes")
    image_size = image_size or int(checkpoint_data.get("image_size", 224))

    log_message(logger, "Prediction data directory: %s", data_dir)
    log_message(logger, "Test image directory: %s", test_image_dir)
    log_message(logger, "Checkpoint: %s", checkpoint_path)
    log_message(logger, "Output path: %s", output_path)
    log_message(logger, "Device: %s | AMP: %s", device, use_amp)
    log_message(logger, "TTA enabled: %s", tta)

    model = build_model(model_name, num_classes)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)

    test_dataset = KaggleImageDataset(
        test_df,
        test_image_dir,
        transform=get_transform(image_size),
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    test_loader = DataLoader(test_dataset, **loader_kwargs)

    id_to_prob = predict(
        model,
        test_loader,
        classes,
        num_classes,
        device,
        use_amp,
        tta=tta,
    )

    submission = test_df[["id"]].copy()
    submission["prob"] = submission["id"].astype(str).map(id_to_prob)

    if submission["prob"].isna().any():
        missing = submission.loc[submission["prob"].isna(), "id"].head(20).tolist()
        raise RuntimeError(f"Missing predictions for ids: {missing}")

    submission["label"] = (submission["prob"] >= threshold).astype(int)
    submission = submission[["id", "prob", "label"]]
    submission.to_csv(output_path, index=False)

    log_message(logger, "Saved submission: %s", output_path)
    log_message(logger, "Saved rows: %d", len(submission))

    return output_path


def main():
    args = parse_args()

    run_prediction(
        data_dir=args.data_dir,
        test_dir=args.test_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_path=args.output_path,
        device_arg=args.device,
        num_workers=args.num_workers,
        tta=not args.no_tta,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
