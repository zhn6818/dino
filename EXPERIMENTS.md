# DINO ResNet50 实验记录

## 环境

- GPU: RTX 3080 Ti (12GB VRAM)
- 数据集: `/root/autodl-tmp/data/jinxiang/jinxiang/` — JLD:1618, JZ:3159, TT:1876, ZZ:1023，共 7676 张
- 固定参数: batch_size=32, local_crops_number=8, lr=0.0005, warmup_epochs=10, optimizer=lars, fp16=true, epochs=100

---

## 实验列表

### 实验 1: Baseline
- **状态**: 训练中
- **可变参数**: wd=0.04→0.4, warmup_temp=0.04, teacher_temp=0.07, warmup_epochs=30, momentum=0.996
- **输出目录**: `dino_resnet50_wd0.04-0.4_t0.04-0.07-30_m0.996`
- **训练 loss**:
- **评估结果**: 未评估

### 实验 2: 低 teacher_temp
- **状态**: 待训练
- **可变参数**: wd=0.04→0.4, warmup_temp=0.04, teacher_temp=0.04, warmup_epochs=30, momentum=0.996
- **输出目录**: `dino_resnet50_wd0.04-0.4_t0.04-0.04-30_m0.996`
- **训练 loss**:
- **评估结果**: 未评估

### 实验 3: 高 momentum
- **状态**: 待训练
- **可变参数**: wd=0.04→0.4, warmup_temp=0.04, teacher_temp=0.07, warmup_epochs=30, momentum=0.9995
- **输出目录**: `dino_resnet50_wd0.04-0.4_t0.04-0.07-30_m0.9995`
- **训练 loss**:
- **评估结果**: 未评估

### 实验 4: 弱正则化
- **状态**: 待训练
- **可变参数**: wd=0.01→0.1, warmup_temp=0.04, teacher_temp=0.07, warmup_epochs=30, momentum=0.996
- **输出目录**: `dino_resnet50_wd0.01-0.1_t0.04-0.07-30_m0.996`
- **训练 loss**:
- **评估结果**: 未评估

### 实验 5: 低温 + 长 warmup
- **状态**: 待训练
- **可变参数**: wd=0.04→0.4, warmup_temp=0.02, teacher_temp=0.05, warmup_epochs=50, momentum=0.996
- **输出目录**: `dino_resnet50_wd0.04-0.4_t0.02-0.05-50_m0.996`
- **训练 loss**:
- **评估结果**: 未评估

---

## 结果汇总

| # | 实验名称 | wd | teacher_temp | momentum | 最终 loss | Top-1 准确率 |
|---|---------|-----|-------------|----------|----------|-------------|
| 1 | Baseline | 0.04→0.4 | 0.07 | 0.996 | - | - |
| 2 | 低 teacher_temp | 0.04→0.4 | 0.04 | 0.996 | - | - |
| 3 | 高 momentum | 0.04→0.4 | 0.07 | 0.9995 | - | - |
| 4 | 弱正则化 | 0.01→0.1 | 0.07 | 0.996 | - | - |
| 5 | 低温+长warmup | 0.04→0.4 | 0.05 | 0.996 | - | - |
