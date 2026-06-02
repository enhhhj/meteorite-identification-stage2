import argparse
import json
from pathlib import Path

import pandas as pd


PROBABILITY_COLUMNS = ("prob", "probability", "pred_prob")
REQUIRED_CONFIG_KEYS = (
    "conv_low",
    "conv_high",
    "swin_low",
    "swin_high",
    "dino_low",
    "dino_high",
    "eff_low",
    "eff_high",
    "w_conv",
    "w_swin",
    "w_dino",
    "w_eff",
    "vote_threshold",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create final ensemble submission from model probabilities.")
    parser.add_argument("--conv", required=True, help="ConvNeXt probability CSV.")
    parser.add_argument("--swin", required=True, help="Swin probability CSV.")
    parser.add_argument("--dino", required=True, help="DINOv2 probability CSV.")
    parser.add_argument("--eff", required=True, help="EfficientNetV2-S probability CSV.")
    parser.add_argument("--config", required=True, help="Ensemble JSON config.")
    parser.add_argument("--out", required=True, help="Output final submission CSV.")
    return parser.parse_args()


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    return config


def read_probability_csv(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{name} CSV not found: {path}")

    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{name} CSV must contain an 'id' column: {path}")

    duplicated = df["id"].astype(str).duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, "id"].astype(str).head(10).tolist()
        raise ValueError(f"{name} CSV contains duplicate ids. Examples: {examples}")

    prob_col = next((col for col in PROBABILITY_COLUMNS if col in df.columns), None)
    if prob_col is None:
        raise ValueError(
            f"{name} CSV must contain one of {PROBABILITY_COLUMNS}. "
            f"Columns found: {list(df.columns)}"
        )

    out = df[["id", prob_col]].copy()
    out["id"] = out["id"].astype(str)
    out = out.rename(columns={prob_col: name})
    out[name] = pd.to_numeric(out[name], errors="raise")
    return out


def validate_id_alignment(frames):
    base_name, base_df = frames[0]
    base_ids = set(base_df["id"])

    for name, df in frames[1:]:
        ids = set(df["id"])
        missing = sorted(base_ids - ids)
        extra = sorted(ids - base_ids)
        if missing or extra:
            raise ValueError(
                f"id alignment failed between {base_name} and {name}: "
                f"{len(missing)} missing, {len(extra)} extra. "
                f"Missing examples: {missing[:10]}; extra examples: {extra[:10]}"
            )


def vote(prob, low, high):
    if prob >= high:
        return 1
    if prob <= low:
        return -1
    return 0


def eff_vote(prob, low, high):
    if prob >= high:
        return -1
    if prob <= low:
        return 1
    return 0


def build_submission(conv, swin, dino, eff, config):
    frames = [("conv", conv), ("swin", swin), ("dino", dino), ("eff", eff)]
    validate_id_alignment(frames)

    merged = conv.merge(swin, on="id").merge(dino, on="id").merge(eff, on="id")
    merged["conv_vote"] = merged["conv"].apply(lambda x: vote(x, config["conv_low"], config["conv_high"]))
    merged["swin_vote"] = merged["swin"].apply(lambda x: vote(x, config["swin_low"], config["swin_high"]))
    merged["dino_vote"] = merged["dino"].apply(lambda x: vote(x, config["dino_low"], config["dino_high"]))
    merged["eff_vote"] = merged["eff"].apply(lambda x: eff_vote(x, config["eff_low"], config["eff_high"]))

    merged["score"] = (
        config["w_conv"] * merged["conv_vote"]
        + config["w_swin"] * merged["swin_vote"]
        + config["w_dino"] * merged["dino_vote"]
        + config["w_eff"] * merged["eff_vote"]
    )
    merged["label"] = (merged["score"] >= config["vote_threshold"]).astype(int)
    return merged[["id", "label"]]


def main():
    args = parse_args()
    config = load_config(args.config)
    conv = read_probability_csv(args.conv, "conv")
    swin = read_probability_csv(args.swin, "swin")
    dino = read_probability_csv(args.dino, "dino")
    eff = read_probability_csv(args.eff, "eff")

    submission = build_submission(conv, swin, dino, eff, config)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved final submission: {output_path} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
