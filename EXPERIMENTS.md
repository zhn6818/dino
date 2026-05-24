# DINO ResNet50 实验记录

## 环境

- GPU: RTX 3080 Ti (12GB VRAM), CPU: 48 核 Xeon, 440GB 内存
- 数据集: `/root/autodl-tmp/data/jinxiang/jinxiang/` — JLD:1618, JZ:3159, TT:1876, ZZ:1023，共 7676 张
- 图像分辨率: 2048x2048 金相图像
- 固定参数: optimizer=sgd, lr=0.03, wd=1e-4(固定), batch_size=16, local_crops_number=4, global_crops_size=448, local_crops_size=192, global_crops_scale=0.3~1.0, local_crops_scale=0.1~0.4, fp16=true, epochs=100

---

## 实验列表

### 实验 1: Baseline
- **状态**: 训练中
- **可变参数**: warmup_temp=0.04, teacher_temp=0.07, warmup_epochs=30, momentum=0.996
- **输出目录**: `dino_resnet50_sgd_lr003_wd1e4_t0.04-0.07-30_m0.996`
- **训练 loss**:
- **评估结果**: 未评估

### 实验 2: 低 teacher_temp
- **状态**: 待训练
- **可变参数**: warmup_temp=0.04, teacher_temp=0.04, warmup_epochs=30, momentum=0.996
- **输出目录**: `dino_resnet50_sgd_lr003_wd1e4_t0.04-0.04-30_m0.996`
- **训练 loss**:
- **评估结果**: 未评估

### 实验 3: 高 momentum
- **状态**: 待训练
- **可变参数**: warmup_temp=0.04, teacher_temp=0.07, warmup_epochs=30, momentum=0.9995
- **输出目录**: `dino_resnet50_sgd_lr003_wd1e4_t0.04-0.07-30_m0.9995`
- **训练 loss**:
- **评估结果**: 未评估

### 实验 4: 低温 + 长 warmup
- **状态**: 待训练
- **可变参数**: warmup_temp=0.02, teacher_temp=0.05, warmup_epochs=50, momentum=0.996
- **输出目录**: `dino_resnet50_sgd_lr003_wd1e4_t0.02-0.05-50_m0.996`
- **训练 loss**:
- **评估结果**: 未评估

### 实验 5: 长 warmup
- **状态**: 待训练
- **可变参数**: warmup_temp=0.04, teacher_temp=0.07, warmup_epochs=50, momentum=0.996
- **输出目录**: `dino_resnet50_sgd_lr003_wd1e4_t0.04-0.07-50_m0.996`
- **训练 loss**:
- **评估结果**: 未评估

---

## 参数调优历史

| 版本 | 参数 | 问题 | 调整 |
|------|------|------|------|
| v1 | lr=0.0005, lars, wd=0.04→0.4, crops=224/96 | 不收敛，lr 太小；裁剪太小丢失金相细节 | 改用官方 ResNet 参数：lr=0.03, sgd, wd=1e-4, crops=448/192 |

---

## 结果汇总

| # | 实验名称 | teacher_temp | momentum | warmup_epochs | 最终 loss | Top-1 准确率 |
|---|---------|-------------|----------|--------------|----------|-------------|
| 1 | Baseline | 0.07 | 0.996 | 30 | - | - |
| 2 | 低 teacher_temp | 0.04 | 0.996 | 30 | - | - |
| 3 | 高 momentum | 0.07 | 0.9995 | 30 | - | - |
| 4 | 低温+长warmup | 0.05 | 0.996 | 50 | - | - |
| 5 | 长 warmup | 0.07 | 0.996 | 50 | - | - |
