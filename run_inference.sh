#!/usr/bin/env bash
set -euo pipefail

# Download model weights from the Baidu Netdisk link in README.md first,
# then place them in this directory with the filenames shown below.
WEIGHT_DIR="${WEIGHT_DIR:-./weights}"
DATA_DIR="${DATA_DIR:-/data/final}"
TEST_DIR="${TEST_DIR:-/data/final/test_images}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

mkdir -p "${OUTPUT_DIR}"

python predict_convnext.py \
  --data_dir "${DATA_DIR}" \
  --test_dir "${TEST_DIR}" \
  --checkpoint "${WEIGHT_DIR}/convnext_tiny_best_model.pth" \
  --image_size 224 \
  --batch_size 32 \
  --out "${OUTPUT_DIR}/convnext_probs.csv" \
  --device auto

python predict_efficientnetv2s.py \
  --data_dir "${DATA_DIR}" \
  --test_dir "${TEST_DIR}" \
  --checkpoint "${WEIGHT_DIR}/efficientv2s_best_model.pth" \
  --image_size 224 \
  --batch_size 32 \
  --out "${OUTPUT_DIR}/efficientnetv2s_probs.csv" \
  --device auto

python predict_dinov2.py \
  --data_dir "${DATA_DIR}" \
  --test_dir "${TEST_DIR}" \
  --checkpoint "${WEIGHT_DIR}/dinov2_best_model.pth" \
  --image_size 224 \
  --batch_size 32 \
  --out "${OUTPUT_DIR}/dinov2_probs.csv" \
  --device auto

python predict_swin.py \
  --data_dir "${DATA_DIR}" \
  --test_dir "${TEST_DIR}" \
  --checkpoint "${WEIGHT_DIR}/swin_ema_best.pth" \
  --image_size 384 \
  --batch_size 16 \
  --out "${OUTPUT_DIR}/swin_probs.csv" \
  --device auto

python ensemble_submit.py \
  --conv "${OUTPUT_DIR}/convnext_probs.csv" \
  --swin "${OUTPUT_DIR}/swin_probs.csv" \
  --dino "${OUTPUT_DIR}/dinov2_probs.csv" \
  --eff "${OUTPUT_DIR}/efficientnetv2s_probs.csv" \
  --config configs/ensemble_config.json \
  --out "${OUTPUT_DIR}/final_submission.csv"
