from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def list_image_files(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_image_dir(data_dir, folder_name):
    data_dir = Path(data_dir)
    folder = Path(folder_name)

    candidates = []
    if folder.is_absolute():
        candidates.append(folder)
    else:
        candidates.extend([data_dir / folder, folder])

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find image directory. Searched: {searched}")


def find_image_path(image_dir, image_id):
    image_dir = Path(image_dir)
    image_id = str(image_id)

    direct = image_dir / image_id
    if direct.is_file():
        return direct

    stem = Path(image_id).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate

    matches = [
        path
        for path in image_dir.rglob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(f"Cannot find image for id={image_id} in {image_dir}")


class KaggleImageDataset(Dataset):
    def __init__(self, df, image_dir, label_col="label", transform=None, labels=None):
        if "id" not in df.columns:
            raise ValueError("Input dataframe must contain an 'id' column.")

        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.label_col = label_col
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = str(self.df.iloc[idx]["id"])
        image_path = find_image_path(self.image_dir, image_id)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        if self.labels is not None:
            return image, torch.as_tensor(self.labels[idx], dtype=torch.long)

        if self.label_col and self.label_col in self.df.columns:
            return image, torch.as_tensor(self.df.iloc[idx][self.label_col], dtype=torch.long)

        return image, image_id
