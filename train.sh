#!/bin/bash

export OMP_NUM_THREADS=4

DATA_PATH="/root/autodl-tmp/data/jinxiang/jinxiang/"
BASE_OUTPUT="/root/autodl-tmp/data/dinomodel/"

# 基于官方 ResNet-50 推荐参数，适配 2048x2048 金相图像
# 官方: sgd, lr=0.03, wd=1e-4(固定), crops_scale=0.14~1/0.05~0.14
# 金相图调整: 增大裁剪分辨率(448/192)，适度调整裁剪比例
COMMON_ARGS="--arch resnet50 \
    --data_path ${DATA_PATH} \
    --batch_size_per_gpu 16 \
    --local_crops_number 4 \
    --epochs 100 \
    --lr 0.03 \
    --warmup_epochs 10 \
    --use_fp16 true \
    --optimizer sgd \
    --weight_decay 1e-4 \
    --weight_decay_end 1e-4 \
    --saveckp_freq 20 \
    --num_workers 16 \
    --global_crops_size 448 \
    --local_crops_size 192 \
    --global_crops_scale 0.3 1.0 \
    --local_crops_scale 0.1 0.4"

# Define parameter groups: each row is one experiment
# Format: warmup_t t t_epochs momentum
PARAM_GROUPS=(
    "0.04|0.07|30|0.996"
    "0.04|0.04|30|0.996"
    "0.04|0.07|30|0.9995"
    "0.02|0.05|50|0.996"
    "0.04|0.07|50|0.996"
)

for params in "${PARAM_GROUPS[@]}"; do
    IFS='|' read -r warmup_t t t_epochs momentum <<< "$params"

    DIR_NAME="dino_resnet50_sgd_lr003_wd1e4_t${warmup_t}-${t}-${t_epochs}_m${momentum}"
    OUTPUT_DIR="${BASE_OUTPUT}/${DIR_NAME}"

    mkdir -p "${OUTPUT_DIR}"

    echo "============================================"
    echo "Training: ${DIR_NAME}"
    echo "============================================"

    torchrun --nproc_per_node=1 --master_port=29500 main_dino.py \
        ${COMMON_ARGS} \
        --output_dir "${OUTPUT_DIR}" \
        --warmup_teacher_temp ${warmup_t} \
        --teacher_temp ${t} \
        --warmup_teacher_temp_epochs ${t_epochs} \
        --momentum_teacher ${momentum} \
        2>&1 | tee "${OUTPUT_DIR}/train.log"

    echo "Finished: ${DIR_NAME}"
    echo ""
done

echo "All experiments completed."
