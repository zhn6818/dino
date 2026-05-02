#!/bin/bash
# DINO 训练脚本
# 硬件：GPU 1-7（7 卡）
#
# 用法：
#   宿主机执行：          bash train.sh docker
#   容器内手动执行：      bash train.sh local
#   后台运行：            nohup bash train.sh docker > train.log 2>&1 &

MODE=${1:-local}
GPUS="1,2,3,4,5,6,7"
NUM_GPUS=7

TRAIN_CMD="source /opt/miniconda3/etc/profile.d/conda.sh && conda activate ai && \
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
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
        --seed 0"

case "$MODE" in
    docker)
        docker exec \
            -w /data2/zhn/code/dino \
            -e NVIDIA_VISIBLE_DEVICES=${GPUS} \
            JHCVTrain \
            bash -c "$TRAIN_CMD"
        ;;
    local)
        export NVIDIA_VISIBLE_DEVICES=${GPUS}
        eval "$TRAIN_CMD"
        ;;
    *)
        echo "用法: bash train.sh [docker|local]"
        echo "  docker - 在 JHCVTrain 容器中训练"
        echo "  local  - 在当前环境（容器内或宿主机）直接训练"
        exit 1
        ;;
esac
