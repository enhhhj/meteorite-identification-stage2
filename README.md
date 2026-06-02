# Meteorite Identification Stage 2

## 项目简介

本项目用于 Meteorite Identification Stage 2 二分类任务，目标是判断测试图片是否为陨石。最终复现方案使用四个模型的测试集概率进行加权投票集成：

- ConvNeXt-Tiny
- EfficientNetV2-S
- DINOv2 ViT-S/14
- Swin Transformer V2-S + EMA

本仓库重点是复现最终测试结果，而不是重新训练出更高分。模型权重不会上传到 GitHub，需要单独下载后放入 `weights/`。

## 文件结构

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── dataset.py
├── inference_utils.py
├── train_convnext.py
├── predict_convnext.py
├── train_efficientnetv2s.py
├── train_efficientnetv2s_teststyle_artifact_aug_nogroup.py
├── predict_efficientnetv2s.py
├── train_dinov2.py
├── predict_dinov2.py
├── train_swin.py
├── predict_swin.py
├── swin.py
├── ensemble_submit.py
├── run_inference.sh
└── configs/
    ├── convnext_best.json
    ├── efficientnetv2s_best.json
    ├── dinov2_best.json
    ├── swin_best.json
    └── ensemble_config.json
```

说明：`train_efficientnetv2s.py` 是清晰入口，调用原始的 EfficientNetV2-S 训练脚本。`swin.py` 保留原始实验代码，`train_swin.py` 是整理后的 argparse 入口。

## 环境安装

建议使用 Python 3.9 或更新版本，并安装 PyTorch 对应 CUDA 版本。

```bash
pip install -r requirements.txt
```

DINOv2 当前代码通过 `torch.hub.load("facebookresearch/dinov2", ...)` 加载骨干网络。首次运行可能需要联网下载，或提前准备好本地 torch hub 缓存。本项目代码未显式使用 `transformers` 或 `xformers`。

## 数据目录要求

默认数据结构如下：

```text
/data/final/
├── train_images/
├── test_images/
├── train_labels.csv
├── sample_submission.csv
└── outputs/
```

`train_labels.csv` 至少包含 `id,label` 两列。预测阶段会优先读取 `sample_submission.csv` 中的 `id` 顺序；如果不存在，则按 `test_images/` 中的图片文件名排序输出。

## 模型权重下载

模型权重未上传至 GitHub，请从以下云盘链接下载，并放入 `weights/` 文件夹：

```text
百度网盘链接：TODO
提取码：TODO

weights/
├── convnext_tiny_best_model.pth
├── efficientv2s_best_model.pth
├── swin_ema_best.pth
└── dinov2_best_model.pth
```

`.gitignore` 已忽略 `*.pth`、`*.pt`、`*.ckpt` 和 `weights/`，避免误提交模型权重。

## 训练命令

ConvNeXt-Tiny：

```bash
python train_convnext.py \
  --data_dir /data/final \
  --epochs 40 \
  --batch_size 32 \
  --lr 0.00003 \
  --weight_decay 0.001 \
  --image_size 224 \
  --patience 8 \
  --run_name convnext_f1_aug_224_seed42
```

EfficientNetV2-S：

```bash
python train_efficientnetv2s.py \
  --data_dir /data/final \
  --epochs 40 \
  --batch_size 32 \
  --lr 0.00003 \
  --weight_decay 0.001 \
  --image_size 224 \
  --tta hflip \
  --run_name efficientnetv2s_nogroup_teststyle_aug_seed42
```

DINOv2 ViT-S/14：

```bash
python train_dinov2.py \
  --data_dir data \
  --epochs 30 \
  --batch_size 128 \
  --lr 0.003 \
  --weight_decay 0.01 \
  --image_size 224 \
  --model_name dinov2_vits14 \
  --concat_layers 2 \
  --train_image_folder train_images_bg_removed \
  --labels_csv train_labels.csv \
  --run_name dinov2_white_cat2_lr3e-3_wd1e-2
```

Swin V2-S + EMA：

```bash
python train_swin.py \
  --data_dir /data/final \
  --epochs 30 \
  --batch_size 16 \
  --lr 0.0001 \
  --weight_decay 0.01 \
  --image_size 384 \
  --patience 10 \
  --run_name swin_v2_s_ema_seed42
```

## 预测命令

ConvNeXt-Tiny：

```bash
python predict_convnext.py \
  --data_dir /data/final \
  --test_dir /data/final/test_images \
  --checkpoint weights/convnext_tiny_best_model.pth \
  --image_size 224 \
  --batch_size 32 \
  --out outputs/convnext_probs.csv \
  --device auto
```

EfficientNetV2-S：

```bash
python predict_efficientnetv2s.py \
  --data_dir /data/final \
  --test_dir /data/final/test_images \
  --checkpoint weights/efficientv2s_best_model.pth \
  --image_size 224 \
  --batch_size 32 \
  --out outputs/efficientnetv2s_probs.csv \
  --device auto
```

DINOv2 ViT-S/14：

```bash
python predict_dinov2.py \
  --data_dir /data/final \
  --test_dir /data/final/test_images \
  --checkpoint weights/dinov2_best_model.pth \
  --image_size 224 \
  --batch_size 32 \
  --out outputs/dinov2_probs.csv \
  --device auto
```

Swin V2-S + EMA：

```bash
python predict_swin.py \
  --data_dir /data/final \
  --test_dir /data/final/test_images \
  --checkpoint weights/swin_ema_best.pth \
  --image_size 384 \
  --batch_size 16 \
  --out outputs/swin_probs.csv \
  --device auto
```

四个预测脚本都会输出：

```text
id,prob,label
```

其中 `prob` 是预测为陨石类的概率，`label` 是按默认阈值 0.5 得到的单模型标签。

## 最终集成

每个模型根据 low/high 阈值转化为 -1、0、+1 三种投票。高于 high 表示正向高置信投票，低于 low 表示负向高置信投票，中间灰区为 0。EfficientNetV2-S 的输出方向在集成阶段经过校准，因此投票方向与其他模型相反。最后根据四个模型的加权投票分数得到最终 `label`。

```python
conv_vote = 1 if conv >= conv_high else (-1 if conv <= conv_low else 0)
swin_vote = 1 if swin >= swin_high else (-1 if swin <= swin_low else 0)
dino_vote = 1 if dino >= dino_high else (-1 if dino <= dino_low else 0)
eff_vote  = -1 if eff >= eff_high else (1 if eff <= eff_low else 0)

score = w_conv * conv_vote + w_swin * swin_vote + w_dino * dino_vote + w_eff * eff_vote
label = 1 if score >= vote_threshold else 0
```

手动运行集成：

```bash
python ensemble_submit.py \
  --conv outputs/convnext_probs.csv \
  --swin outputs/swin_probs.csv \
  --dino outputs/dinov2_probs.csv \
  --eff outputs/efficientnetv2s_probs.csv \
  --config configs/ensemble_config.json \
  --out outputs/final_submission.csv
```

## 一键复现

先下载权重并放入 `weights/`，再运行：

```bash
bash run_inference.sh
```

可以通过环境变量覆盖默认路径：

```bash
WEIGHT_DIR=./weights DATA_DIR=/data/final TEST_DIR=/data/final/test_images bash run_inference.sh
```

## 最终输出

最终提交文件为：

```text
outputs/final_submission.csv
```

格式为：

```text
id,label
```
