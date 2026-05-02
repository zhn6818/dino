#!/bin/bash
# DINO 训练脚本
# 硬件：单机 8 GPU（torchrun --nproc_per_node=8）
# 数据集：jinxiang（JLD/JZ/TT/ZZ 等）
# nohup bash train.sh > train.log 2>&1 &

torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --data_path /data2/zhn/code/data/jinxiang/ \
    --output_dir ./dino_output \
    --epochs 1000 \
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
