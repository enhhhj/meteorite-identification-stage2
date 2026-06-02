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
from train_dinov2 import build_model, get_transforms, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Predict DINOv2 classifier probabilities.")
    parser.add_argument("--data_dir", type=str, default="/data/final")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="weights/dinov2_best_model.pth")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", type=str, default="outputs/dinov2_probs.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    parser.add_argument("--concat_layers", type=int, default=None)
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


def run_prediction(
    data_dir,
    checkpoint,
    test_dir=None,
    batch_size=32,
    image_size=224,
    output_path="outputs/dinov2_probs.csv",
    device_arg="auto",
    num_workers=4,
    logger=None,
    threshold=0.5,
    tta="none",
    concat_layers=None,
):
    device = resolve_device(device_arg)
    use_amp = device.type == "cuda"

    checkpoint_data = load_checkpoint(checkpoint, map_location="cpu")
    classes = checkpoint_data.get("classes")
    num_classes = int(checkpoint_data.get("num_classes", len(classes) if classes is not None else 2))
    model_name = checkpoint_data.get("model_name", "dinov2_vits14")
    image_size = int(checkpoint_data.get("image_size", image_size))
    concat_layers = int(concat_layers or checkpoint_data.get("concat_layers", 2))
    positive_index = positive_class_index(classes, num_classes)

    _, eval_transform = get_transforms(image_size)
    test_df, test_image_dir = make_test_df(data_dir, test_dir)
    dataset = KaggleImageDataset(test_df, test_image_dir, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        concat_layers=concat_layers,
    )
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    model.to(device)

    image_ids, probabilities = predict_probabilities(
        model=model,
        loader=loader,
        positive_index=positive_index,
        device=device,
        use_amp=use_amp,
        tta=tta,
    )
    output_path = save_probability_csv(image_ids, probabilities, output_path, threshold=threshold)
    print(f"Saved DINOv2 probabilities: {output_path}")
    return output_path


def main():
    args = parse_args()
    run_prediction(
        data_dir=args.data_dir,
        test_dir=args.test_dir,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        output_path=args.out,
        device_arg=args.device,
        num_workers=args.num_workers,
        threshold=args.threshold,
        tta=args.tta,
        concat_layers=args.concat_layers,
    )


if __name__ == "__main__":
    main()
