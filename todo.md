# DINO 高分辨率细节特征自监督训练优化

## 场景
- 输入图像: 2048×2048
- 图像内容: 细节特征，最小 5×5 像素
- 硬件: 8× RTX 2080 SUPER (8GB)

## 核心问题
默认 DINO 参数下，5×5 特征在任何裁剪中都不到 1 个 patch，几乎完全丢失：
- 默认全局裁剪 (224, scale=0.4): 特征仅 1.4 px
- 默认局部裁剪 (96, scale=0.05): 特征仅 4.7 px

优化后裁剪参数下的特征保留情况：
| 设定 | crop | scale | 特征像素 |
|------|------|-------|---------|
| 全局裁剪(小scale) | 512 | 0.1 | 12.5 px ✓ |
| 全局裁剪(大scale) | 512 | 0.3 | 4.2 px △ |
| 局部裁剪(小scale) | 256 | 0.05 | 12.5 px ✓ |
| 局部裁剪(大scale) | 256 | 0.15 | 4.2 px △ |

## 优化方向

### 裁剪策略重塑
- [x] 放弃传统"全局+局部"范式，改为"多尺度密集裁剪"
- [x] global_crop_size 提到 512，缩小 scale 范围至 0.1-0.3
- [x] local_crop_size 提到 256，scale 范围 0.05-0.15
- [ ] local_crops_number: 先用 8 验证流程，再逐步增加到 16
- [ ] 本质：把 2048×2048 图像当作"地图"来密集采样

### Patch size（ViT 场景，后续迭代）
- [ ] patch_size=8 最低要求
- [ ] patch_size=4 最理想但计算量再翻 4 倍
- [ ] 需要 `--use_fp16 false` 保证小 patch 训练稳定性

### 架构选择
- [ ] **V1: ResNet50** — CNN 不存在 patch 量化问题，各层自然保留多尺度信息，适合快速验证
- [ ] V2: XCiT（线性注意力复杂度）— 天然适合高分辨率，后续探索
- [ ] V3: ViT + patch_size=4 + 密集裁剪 — 计算资源允许时再尝试

## V1 执行计划: ResNet50 训练

### ResNet50 优势（针对本场景）
- CNN 通过卷积核逐层提取特征，不存在 patch 量化问题
- 浅层 feature map 分辨率高（如 conv1 输出 1024×1024），天然能捕获 5×5 细节
- 深层 feature map 提供全局语义，形成多尺度表达
- 内存友好，2048×2048 输入也可承受

### 训练命令
```bash
# ResNet50 + 高分辨率细节特征优化（docker 模式）
nohup bash train.sh \
    --mode docker \
    --arch resnet50 \
    --epochs 100 \
    --batch_size_per_gpu 4 \
    --global_crop_size 512 \
    --local_crop_size 256 \
    --global_crops_scale "0.1 0.3" \
    --local_crops_scale "0.05 0.15" \
    --local_crops_number 8 \
    --optimizer lars \
    --use_fp16 true \
    > train_resnet50_hr.log 2>&1 &

# 如果 OOM，降低 batch_size：
#   --batch_size_per_gpu 2

# 如果显存充足，可增加 local_crops_number：
#   --local_crops_number 16
```

### 参数说明
| 参数 | 值 | 说明 |
|------|---|------|
| arch | resnet50 | CNN backbone，无 patch 量化问题 |
| global_crop_size | 512 | 全局裁剪 512×512（默认 224） |
| local_crop_size | 256 | 局部裁剪 256×256（默认 96） |
| global_crops_scale | 0.1 0.3 | 缩小 scale 保留细节（默认 0.4 1.0） |
| local_crops_scale | 0.05 0.15 | 缩小 scale 保留细节（默认 0.05 0.4） |
| local_crops_number | 8 | 密集采样（默认 4） |
| batch_size_per_gpu | 4 | 8GB 显存保守估计 |
| optimizer | lars | CNN 推荐优化器 |
