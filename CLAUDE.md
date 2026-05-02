# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

DINO（Self-DIstillation with NO labels）——Meta AI（Facebook Research）提出的面向 Vision Transformer 的自监督学习方法。通过无标签的知识蒸馏训练学生-教师网络，产出的特征可用于 k-NN 分类、线性评估、图像检索、副本检测和视频分割。

## 常用命令

### 训练
```bash
# 单节点 8 GPU（标准配置）
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
    --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/output

# 通过 Slurm/submitit 多节点训练
python run_with_submitit.py --nodes 2 --ngpus 8 --arch vit_small \
    --data_path /path/to/imagenet/train --output_dir /path/to/output

# 查看完整参数列表
python main_dino.py --help
```

### 评估
```bash
# ImageNet k-NN 分类（1 GPU）
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --data_path /path/to/imagenet

# 线性分类（8 GPU）
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /path/to/imagenet

# 使用指定 checkpoint 做线性评估
python eval_linear.py --evaluate --arch vit_small --patch_size 16 --data_path /path/to/imagenet/train
```

### 可视化
```bash
python visualize_attention.py  # 自注意力图
python video_generation.py --pretrained_weights dino_deitsmall8_pretrain.pth \
    --input_path input/video.mp4 --output_path output/ --fps 25
```

## 架构

### 核心训练流程 (main_dino.py)
- **DINOLoss**：学生 softmax 输出与教师 centered-sharpened softmax 输出之间的交叉熵。教师温度采用 warmup 调度，center 通过分布式 EMA 更新。
- **DataAugmentationDINO**：生成 2 个全局裁剪（224x224）+ N 个局部裁剪（96x96），含翻转/颜色抖动/高斯模糊/曝光处理。
- **训练循环**：学生网络处理所有裁剪，教师网络仅处理 2 个全局裁剪。教师通过 EMA 从学生更新（余弦动量调度 0.996→1）。

### 模型 (vision_transformer.py)
- **VisionTransformer**：Patch 嵌入 → [CLS] token + 位置编码 → N 个 Transformer 块 → LayerNorm，返回 CLS token 特征。
- **DINOHead**：MLP（3 层，hidden_dim=2048，bottleneck_dim=256）→ L2 归一化 → 权重归一化线性投影至 `out_dim`（默认 65536）。
- **工厂函数**：`vit_tiny`（192 维）、`vit_small`（384 维）、`vit_base`（768 维），均为 12 个 block。
- **MultiCropWrapper**（位于 utils.py）：处理不同分辨率裁剪的 backbone 前向传播，拼接特征后送入 head。

### 关键工具函数 (utils.py)
- `cosine_scheduler`：用于学习率、权重衰减、教师动量的余弦调度。
- `MultiCropWrapper`：将相同分辨率的输入批量处理以提高效率。
- `LARS` 优化器：用于卷积网络训练。
- `PCA` 类与 `compute_map`：用于检索/副本检测评估。
- `load_pretrained_weights` / `load_pretrained_linear_weights`：自动从 fbaipublicfiles 下载 DINO 参考权重。

### 评估脚本
- `eval_knn.py`：冻结特征上的 k-NN 分类。
- `eval_linear.py`：在冻结 backbone 上训练有监督线性分类器。
- `eval_image_retrieval.py`：在 revisited Oxford/Paris 基准上进行图像检索。
- `eval_copy_detection.py`：在 Copydays 数据集上进行副本检测。
- `eval_video_segmentation.py`：DAVIS 2017 视频目标分割。

### 模型加载 (hubconf.py)
PyTorch Hub 入口，如 `torch.hub.load('facebookresearch/dino:main', 'dino_vits16')` 等。

## 支持的网络架构
- ViT（vit_tiny/small/base，patch_size 为 8 或 16）
- XCiT（通过 `facebookresearch/xcit` hub 加载）
- torchvision 卷积网络（如 ResNet-50）

## 依赖
PyTorch、torchvision、PIL、numpy。可选：submitit（多节点训练）、timm。开发环境为 PyTorch 1.7.1、CUDA 11.0、torchvision 0.8.2、Python 3.6。
