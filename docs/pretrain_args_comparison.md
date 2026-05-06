# DINO 预训练模型参数对比


| 参数                             | deit_small16 | deit_small8 | vit_base16   | vit_base8    | resnet50     |
| ------------------------------ | ------------ | ----------- | ------------ | ------------ | ------------ |
| **arch**                       | vit_small    | vit_small   | vit_base     | vit_base     | resnet50     |
| **patch_size**                 | 16           | 8           | 16           | 8            | —            |
| **out_dim**                    | 65536        | 65536       | 65536        | 65536        | 60000        |
| **norm_last_layer**            | false        | false       | true         | true         | true         |
| **warmup_teacher_temp**        | 0.04         | 0.04        | 0.04         | 0.03         | 0.04         |
| **teacher_temp**               | 0.07         | 0.07        | 0.07         | 0.07         | 0.07         |
| **warmup_teacher_temp_epochs** | 30           | 30          | 50           | 50           | 50           |
| **weight_decay**               | 0.04         | 0.04        | 0.04         | 0.04         | 1e-6         |
| **weight_decay_end**           | 0.4          | 0.4         | 0.4          | 0.4          | 1e-6         |
| **clip_grad**                  | 0            | 3.0         | 0.3          | 3.0          | 0            |
| **batch_size_per_gpu**         | 64           | 16          | 32           | 6            | 51           |
| **epochs**                     | 800          | 800         | 400          | 300          | 800          |
| **freeze_last_layer**          | 1            | 1           | 3            | 3            | 1            |
| **lr**                         | 0.0005       | 0.0005      | 0.00075      | 0.0005       | 0.3          |
| **warmup_epochs**              | 10           | 10          | 10           | 10           | 10           |
| **min_lr**                     | 1e-5         | 1e-6        | 2e-6         | 2e-6         | 0.0048       |
| **global_crops_scale**         | [0.25, 1.0]  | [0.4, 1.0]  | [0.25, 1.0]  | [0.25, 1.0]  | [0.14, 1.0]  |
| **local_crops_scale**          | [0.05, 0.25] | [0.05, 0.4] | [0.05, 0.25] | [0.05, 0.25] | [0.05, 0.14] |
| **local_crops_number**         | 10           | 10          | 10           | 10           | 6            |
| **optimizer**                  | adamw        | adamw       | adamw        | adamw        | lars         |
| **momentum_teacher**           | 0.996        | 0.996       | 0.996        | 0.996        | 0.996        |
| **use_bn_in_head**             | false        | false       | false        | false        | true         |
| **drop_path_rate**             | 0.1          | 0.1         | 0.1          | 0.1          | —            |
| **world_size (GPUs)**          | 16           | 64          | 32           | 176          | 80           |
| **nodes**                      | 2            | 8           | 4            | 22           | —            |


## 关键差异总结

- **ResNet50** 与 ViT 系列差异最大：使用 LARS 优化器（vs AdamW）、大 1000 倍的权重衰减、大 600 倍的学习率、更小的 `out_dim`（60000 vs 65536）、更少的局部裁剪（6 vs 10），且 head 中使用 BatchNorm。
- **patch_size=8** 的模型比 patch_size=16 的需要更少显存（batch_size 更小），但用更多 GPU 训练（vit_base8 用了 176 块 GPU）。
- **小模型**（deit_small）训练 800 epochs，大模型更少（vit_base16: 400, vit_base8: 300）。
- **教师温度 warmup**：小模型 30 epochs，大模型和 ResNet50 为 50 epochs。

> 数据来源：[https://dl.fbaipublicfiles.com/dino/](https://dl.fbaipublicfiles.com/dino/)

