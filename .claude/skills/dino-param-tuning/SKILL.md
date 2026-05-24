---
name: dino-param-tuning
description: DINO ResNet50 超参数对比训练与评估流程。当用户提到 DINO 训练、参数对比、超参数调优、评估模型、ResNet50 自监督训练、或需要恢复训练计划时触发此 skill。
---

# DINO ResNet50 参数对比训练流程

## 项目状态

当前处于**参数对比训练阶段**。已完成 train.sh 编写，尚未开始训练。

## 环境

- GPU: RTX 3080 Ti (12GB VRAM)
- 数据集: `/root/autodl-tmp/data/jinxiang/jinxiang/` — 4 类 (JLD:1618, JZ:3159, TT:1876, ZZ:1023)，共 7676 张图
- 模型: ResNet50 + DINO 自监督框架
- 优化器: LARS + FP16 混合精度
- batch_size_per_gpu: 32, local_crops_number: 8

## 第一步：训练（train.sh）

运行 `bash train.sh` 执行 5 组超参数对比实验。

### 5 组实验设计

| # | weight_decay | wd_end | warmup_teacher_temp | teacher_temp | warmup_epochs | momentum | 对比目的 |
|---|---|---|---|---|---|---|---|
| 1 | 0.04→0.4 | | 0.04 | 0.07 | 30 | 0.996 | Baseline |
| 2 | 0.04→0.4 | | 0.04 | 0.04 | 30 | 0.996 | vs 1: 教师 softmax 更尖锐 |
| 3 | 0.04→0.4 | | 0.04 | 0.07 | 30 | 0.9995 | vs 1: 教师 EMA 更新更慢 |
| 4 | 0.01→0.1 | | 0.04 | 0.07 | 30 | 0.996 | vs 1: 弱正则化 |
| 5 | 0.04→0.4 | | 0.02 | 0.05 | 50 | 0.996 | vs 1: 低温+更长 warmup |

输出目录: `/root/autodl-tmp/dino_resnet50_wd{wd}-{wd_end}_t{warmup_t}-{t}-{t_epochs}_m{momentum}/`
每个目录下包含: `checkpoint.pth`, `checkpoint{epoch}.pth`, `log.txt`, `train.log`

### 修改实验参数

编辑 `train.sh` 中的 `PARAM_GROUPS` 数组即可增减或调整参数组合，格式：
```
"weight_decay|weight_decay_end|warmup_teacher_temp|teacher_temp|warmup_teacher_temp_epochs|momentum_teacher"
```

## 第二步：数据划分（训练完成后）

DINO 自监督训练使用全部 7676 张图（不用标签）。评估阶段需要划分 train/val：

```python
# 将 /root/autodl-tmp/data/jinxiang/jinxiang/ 划分为
# /root/autodl-tmp/data/jinxiang_split/train/  (80%)
# /root/autodl-tmp/data/jinxiang_split/val/   (20%)
# 保持各类别比例一致（stratified split）
```

划分后的目录结构：
```
jinxiang_split/
├── train/
│   ├── JLD/
│   ├── JZ/
│   ├── TT/
│   └── ZZ/
└── val/
    ├── JLD/
    ├── JZ/
    ├── TT/
    └── ZZ/
```

## 第三步：线性评估（eval_linear.py）

对每个模型的 teacher checkpoint 做线性评估，比较 Top-1 准确率：

```bash
torchrun --nproc_per_node=1 eval_linear.py \
    --arch resnet50 \
    --pretrained_weights /root/autodl-tmp/{模型目录}/checkpoint.pth \
    --checkpoint_key teacher \
    --data_path /root/autodl-tmp/data/jinxiang_split \
    --num_labels 4 \
    --output_dir /root/autodl-tmp/{模型目录}/eval_linear \
    --epochs 100 \
    --lr 0.001 \
    --batch_size_per_gpu 64
```

## 第四步：结果对比

汇总 5 个模型的 Top-1 准确率，生成对比表格，确定最优参数组合。

## 已做的代码修改

`main_dino.py` 中已移除 XCiT 相关代码（`torch.hub.list("facebookresearch/xcit:main")` 调用），避免缺少 `timm` 导致报错。如果需要恢复 XCiT 支持，需要 `pip install timm`。
