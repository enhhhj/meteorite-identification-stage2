from pathlib import Path

import pandas as pd
import torch

from dataset import find_image_path, list_image_files, resolve_image_dir


PROBABILITY_COLUMNS = ("prob", "probability", "pred_prob")


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device, but CUDA is not available.")

    return device


def load_checkpoint(path, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_output_path(output_path):
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".csv":
        output_path = output_path / "probs.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def make_test_df(data_dir, test_dir=None):
    data_dir = Path(data_dir)
    test_image_dir = Path(test_dir) if test_dir else resolve_image_dir(data_dir, "test_images")

    image_files = list_image_files(test_image_dir)
    if not image_files:
        raise FileNotFoundError(f"No test images found in {test_image_dir}")

    sample_path = data_dir / "sample_submission.csv"
    if sample_path.is_file():
        sample_df = pd.read_csv(sample_path)
        if "id" not in sample_df.columns:
            raise ValueError("sample_submission.csv must contain an 'id' column.")

        ids = sample_df["id"].astype(str).tolist()
        missing = []
        for image_id in ids:
            try:
                find_image_path(test_image_dir, image_id)
            except FileNotFoundError:
                missing.append(image_id)

        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} ids from sample_submission.csv were not found in "
                f"{test_image_dir}. First missing ids: {preview}"
            )

        return sample_df[["id"]].copy(), test_image_dir

    ids = [path.name for path in image_files]
    return pd.DataFrame({"id": ids}), test_image_dir


def positive_class_index(classes, num_classes):
    if classes is not None:
        normalized = [str(item.item() if hasattr(item, "item") else item) for item in classes]
        if "1" in normalized:
            return normalized.index("1")

    if int(num_classes) == 2:
        return 1

    raise ValueError(
        "Cannot infer positive class index. Checkpoint classes must contain label 1 "
        "or the model must be binary."
    )


def save_probability_csv(image_ids, probabilities, output_path, threshold=0.5):
    output_path = normalize_output_path(output_path)
    df = pd.DataFrame({"id": [str(x) for x in image_ids], "prob": probabilities})
    df["label"] = (df["prob"] >= threshold).astype(int)
    df.to_csv(output_path, index=False)
    return output_path
