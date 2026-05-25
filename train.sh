#!/bin/bash

export OMP_NUM_THREADS=4

# ============ 可配置参数 ============
GPUS="3,4,5"                        # 使用的 GPU 编号
DATA_PATH="/data1/code/cv_workspace/datasets/dino_data/"
BASE_OUTPUT="/data1/code/dino/output2"
MASTER_PORT=29500
# ===================================

GPU_COUNT=$(echo $GPUS | tr ',' '\n' | wc -l)

COMMON_ARGS="--arch resnet50 \
    --data_path ${DATA_PATH} \
    --batch_size_per_gpu 16 \
    --local_crops_number 4 \
    --epochs 500 \
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

DIR_NAME="dino_resnet50_sgd_lr003_wd1e4_t0.04-0.07-50_m0.996"
OUTPUT_DIR="${BASE_OUTPUT}/${DIR_NAME}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Training: ${DIR_NAME} on GPU ${GPUS} (${GPU_COUNT} cards)"
echo "============================================"

CUDA_VISIBLE_DEVICES=${GPUS} \
torchrun --nproc_per_node=${GPU_COUNT} --master_port=${MASTER_PORT} main_dino.py \
    ${COMMON_ARGS} \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_teacher_temp 0.04 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 50 \
    --momentum_teacher 0.996 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training completed."
