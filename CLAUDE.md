# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DINO (Self-Supervised Vision Transformers) — PyTorch implementation of self-distillation with no labels. Trains vision backbones (ViT, ResNet, XCiT) using a student-teacher framework where the teacher is an EMA of the student. Paper: "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021).

## Training

Single GPU:
```bash
torchrun --nproc_per_node=1 --master_port=29500 main_dino.py \
    --arch resnet50 --data_path /path/to/data/ --output_dir /path/to/output
```

Multi-node (via Slurm + submitit):
```bash
python run_with_submitit.py --nodes 2 --ngpus 8 --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

Batch training with parameter sweeps:
```bash
bash train.sh
```

### Architecture-specific recommendations
- **ViT**: Use `adamw` optimizer (default). FP16 safe with patch_size >= 16.
- **ResNet**: Use `lars` or `sgd` optimizer. Adjust `--global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14` for better performance.
- **XCiT**: Requires `timm` package. Removed from argument choices to avoid import errors when `timm` is not installed.

### Key hyperparameters for tuning
- `--weight_decay` / `--weight_decay_end`: Cosine schedule from start to end (ViT: 0.04→0.4, ResNet: 1e-4→1e-4 constant)
- `--warmup_teacher_temp` / `--teacher_temp` / `--warmup_teacher_temp_epochs`: Teacher temperature schedule (lower = sharper distribution)
- `--momentum_teacher`: EMA rate (0.996 default, use 0.9995 for small batches)
- `--out_dim`: Projection head output dimension (65536 default)
- `--local_crops_number`: Number of small crops (8 default, reduce to save memory)

## Evaluation

All eval scripts require distributed launch. The data path must contain `train/` and `val/` subdirectories with ImageFolder structure.

**k-NN evaluation** (no training, fast):
```bash
torchrun --nproc_per_node=1 eval_knn.py --arch resnet50 \
    --pretrained_weights /path/to/checkpoint.pth --checkpoint_key teacher \
    --data_path /path/to/imagenet
```

**Linear evaluation** (train linear classifier on frozen features):
```bash
torchrun --nproc_per_node=1 eval_linear.py --arch resnet50 \
    --pretrained_weights /path/to/checkpoint.pth --checkpoint_key teacher \
    --data_path /path/to/imagenet --num_labels 4 --output_dir /path/to/eval_output
```

Note: For custom datasets, set `--num_labels` to the number of classes.

**Other eval scripts**: `eval_image_retrieval.py`, `eval_copy_detection.py`, `eval_video_segmentation.py`

## Architecture

### Core files
- `main_dino.py` — Training entry point. Contains `DINOLoss`, `DataAugmentationDINO`, and training loop. Loads model by arch name: ViT from `vision_transformer.py`, ResNet from torchvision, XCiT from torch hub.
- `vision_transformer.py` — ViT implementations (tiny/small/base) with `DINOHead` projection MLP.
- `utils.py` — Distributed training setup, `MultiCropWrapper` (handles multi-resolution forward pass), data augmentation helpers (`GaussianBlur`, `Solarization`), LARS optimizer, checkpoint/logging utilities.
- `hubconf.py` — PyTorch Hub model definitions for loading pretrained weights.

### Model loading flow
`main_dino.py` line ~161: arch name is matched against vits.__dict__ → torchvision_models.__dict__. For torchvision models (e.g. resnet50), `embed_dim` is extracted from `model.fc.weight.shape[1]`, and `model.fc` is replaced by the DINOHead inside MultiCropWrapper.

### Checkpoint structure
- `checkpoint.pth` — Latest checkpoint (student + teacher + optimizer + loss state)
- `checkpoint{epoch}.pth` — Periodic saves (controlled by `--saveckp_freq`)
- `log.txt` — Per-epoch training stats (JSON lines)
- Key to load backbone: `checkpoint_key="teacher"` gives the best features

### Data format
Expects ImageFolder layout: `data_path/class_name/image.jpg`. The `DataAugmentationDINO` class generates 2 global crops (224x224) + N local crops (96x96) per image.

## Current Setup

- **Dataset**: `/root/autodl-tmp/data/jinxiang/jinxiang/` — 4 classes (JLD, JZ, TT, ZZ), ~7676 images
- **GPU**: RTX 3080 Ti (12GB VRAM)
- **Model**: ResNet50 with LARS optimizer, FP16 training
- **Batch training script**: `train.sh` runs 5 hyperparameter experiments with automatic output directory naming (`dino_resnet50_wd{wd}-{wd_end}_t{warmup_t}-{t}-{t_epochs}_m{momentum}`)
- **XCiT code paths removed** from `main_dino.py` (args choices and model loading) due to missing `timm` dependency
