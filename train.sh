#!/bin/bash
# DINO 训练脚本
# 硬件：单张 RTX 2060 Super (8GB)
# 数据集：/data1/zhn/jinxiang（JLD/JZ/TT/ZZ，共约 7676 张图）

torchrun \
    --nproc_per_node=1 \
    --master_port=29502 \
    main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --data_path /data1/zhn/jinxiang \
    --output_dir /data1/zhn/dino_output \
    --epochs 100 \
    --batch_size_per_gpu 16 \
    --local_crops_number 4 \
    --use_fp16 true \
    --optimizer adamw \
    --lr 0.0005 \
    --warmup_epochs 10 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --saveckp_freq 20 \
    --num_workers 4 \
    --seed 0
