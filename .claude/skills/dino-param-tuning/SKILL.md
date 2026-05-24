---
name: dino-param-tuning
description: DINO ResNet50 超参数对比训练与评估流程。当用户提到 DINO 训练、参数对比、超参数调优、评估模型、ResNet50 自监督训练、金相图像、或需要恢复训练计划时触发此 skill。
---

# DINO ResNet50 参数对比训练流程

## 项目状态

当前处于**参数对比训练阶段**。基于官方 ResNet-50 推荐参数，适配 2048x2048 金相图像，正在训练中。

## 环境

- GPU: RTX 3080 Ti (12GB VRAM)
- CPU: Intel Xeon Silver 4214R (48 核), 440GB 内存
- 数据集: `/root/autodl-tmp/data/jinxiang/jinxiang/` — 4 类 (JLD:1618, JZ:3159, TT:1876, ZZ:1023)，共 7676 张图
- 图像分辨率: **2048x2048** 金相图像（晶界、相结构、夹杂物等微观特征需要高分辨率）
- 模型: ResNet50 + DINO 自监督框架
- 输出目录: `/root/autodl-tmp/data/dinomodel/`

## 代码修改记录

1. **移除 XCiT 依赖** — `main_dino.py` 中移除了 `torch.hub.list("facebookresearch/xcit:main")` 调用和 XCiT 模型加载分支，避免缺少 `timm` 导致报错
2. **新增裁剪尺寸参数** — `main_dino.py` 新增 `--global_crops_size` 和 `--local_crops_size` 命令行参数（原硬编码 224/96），`DataAugmentationDINO` 类相应修改

## 训练参数设计

### 固定参数（基于官方 ResNet-50 推荐）

官方推荐: `--arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4`

| 参数 | 值 | 说明 |
|------|------|------|
| `--optimizer` | sgd | 官方推荐 ResNet 用 SGD |
| `--lr` | 0.03 | 官方推荐值（ViT 用 0.0005，ResNet 需要大 lr） |
| `--weight_decay` | 1e-4 | 官方推荐（固定，不做 cosine schedule） |
| `--weight_decay_end` | 1e-4 | 固定值 |
| `--global_crops_size` | 448 | 金相图适配（原 224 太小，2048→448 保留更多细节） |
| `--local_crops_size` | 192 | 金相图适配（原 96 太小） |
| `--global_crops_scale` | 0.3~1.0 | 折中（官方 0.14~1.0，金相图提高下限保留上下文） |
| `--local_crops_scale` | 0.1~0.4 | 折中（官方 0.05~0.14，金相图需要更大局部区域） |
| `--batch_size_per_gpu` | 16 | 大裁剪尺寸更吃显存 |
| `--local_crops_number` | 4 | 配合大裁剪减少显存压力 |
| `--epochs` | 100 | |
| `--warmup_epochs` | 10 | |
| `--use_fp16` | true | |
| `--num_workers` | 16 | 充分利用 48 核 CPU |
| `--saveckp_freq` | 20 | |

### 5 组对比实验（变量：temperature 和 momentum）

| # | warmup_teacher_temp | teacher_temp | warmup_epochs | momentum | 对比目的 |
|---|---|---|---|---|---|
| 1 | 0.04 | 0.07 | 30 | 0.996 | Baseline |
| 2 | 0.04 | 0.04 | 30 | 0.996 | vs 1: 教师 softmax 更尖锐 |
| 3 | 0.04 | 0.07 | 30 | 0.9995 | vs 1: 教师 EMA 更新更慢 |
| 4 | 0.02 | 0.05 | 50 | 0.996 | vs 1: 低温+更长 warmup |
| 5 | 0.04 | 0.07 | 50 | 0.996 | vs 1: 仅延长 warmup |

输出目录命名: `dino_resnet50_sgd_lr003_wd1e4_t{warmup_t}-{t}-{t_epochs}_m{momentum}`

### 修改实验参数

编辑 `train.sh` 中的 `PARAM_GROUPS` 数组，格式：
```
"warmup_teacher_temp|teacher_temp|warmup_teacher_temp_epochs|momentum_teacher"
```

## 第一步：训练

```bash
bash train.sh
```

## 第二步：数据划分（训练完成后）

DINO 自监督训练使用全部 7676 张图（不用标签）。评估阶段需要划分 train/val：

```python
# 将 /root/autodl-tmp/data/jinxiang/jinxiang/ 划分为
# /root/autodl-tmp/data/jinxiang_split/train/  (80%)
# /root/autodl-tmp/data/jinxiang_split/val/   (20%)
# 保持各类别比例一致（stratified split）
```

## 第三步：线性评估（eval_linear.py）

对每个模型的 teacher checkpoint 做线性评估，比较 Top-1 准确率：

```bash
torchrun --nproc_per_node=1 eval_linear.py \
    --arch resnet50 \
    --pretrained_weights /root/autodl-tmp/data/dinomodel/{模型目录}/checkpoint.pth \
    --checkpoint_key teacher \
    --data_path /root/autodl-tmp/data/jinxiang_split \
    --num_labels 4 \
    --output_dir /root/autodl-tmp/data/dinomodel/{模型目录}/eval_linear \
    --epochs 100 \
    --lr 0.001 \
    --batch_size_per_gpu 64
```

## 第四步：结果对比

汇总 5 个模型的 Top-1 准确率，生成对比表格，确定最优参数组合。结果记录在 `/root/code/dino/EXPERIMENTS.md`。

## 参数调优历史

| 版本 | 主要问题 | 调整 |
|------|---------|------|
| v1 | lr=0.0005 (ViT 参数)，不收敛 | 改用官方 ResNet 参数 lr=0.03, sgd, wd=1e-4 |
| v1 | 裁剪 224/96 对 2048x2048 金相图丢失太多细节 | 增大至 448/192，调整裁剪比例 |
| v1 | optimizer=lars | 改为 sgd（官方推荐） |
| v1 | num_workers=4, batch=32 | 调为 num_workers=16, batch=16（大图更吃显存） |
