import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KaggleImageDataset
from inference_utils import (
    load_checkpoint,
    make_test_df,
    positive_class_index,
    save_probability_csv,
)
from train_efficientnetv2s_teststyle_artifact_aug_nogroup import (
    build_model,
    get_transforms,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict EfficientNetV2-S probabilities.")
    parser.add_argument("--data_dir", type=str, default="/data/final")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="weights/efficientv2s_best_model.pth")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", type=str, default="outputs/efficientnetv2s_probs.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    return parser.parse_args()


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
    classes = checkpoint.get("classes")
    num_classes = int(checkpoint.get("num_classes", len(classes) if classes is not None else 2))
    model_name = checkpoint.get("model_name", "efficientnet_v2_s")
    image_size = int(checkpoint.get("image_size", args.image_size))
    positive_index = positive_class_index(classes, num_classes)

    _, eval_transform = get_transforms(image_size)
    test_df, test_image_dir = make_test_df(args.data_dir, args.test_dir)
    dataset = KaggleImageDataset(test_df, test_image_dir, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    image_ids, probabilities = predict_probabilities(
        model=model,
        loader=loader,
        positive_index=positive_index,
        device=device,
        use_amp=use_amp,
        tta=args.tta,
    )
    output_path = save_probability_csv(image_ids, probabilities, args.out, threshold=args.threshold)
    print(f"Saved EfficientNetV2-S probabilities: {output_path}")


if __name__ == "__main__":
    main()
