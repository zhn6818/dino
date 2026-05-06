#!/bin/bash

# DINO ResNet50 训练脚本
# 参考: https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/args.txt

DATA_PATH=/data2/zhn/code/data/jinxiang/
OUTPUT_DIR=./output/resnet50
BATCH_SIZE_PER_GPU=4
LOCAL_CROPS_NUMBER=6
GLOBAL_CROP_SIZE=512
LOCAL_CROP_SIZE=256
GLOBAL_CROPS_SCALE="0.05 0.30"
LOCAL_CROPS_SCALE="0.01 0.08"
BASE_LR=0.3
MIN_LR_RATIO=0.001

# 检测可用 GPU 数量
NUM_GPUS=$(nvidia-smi -L | wc -l)
EFFECTIVE_MAX_LR=$(awk "BEGIN {printf \"%.8g\", ${BASE_LR} * (${BATCH_SIZE_PER_GPU} * ${NUM_GPUS}) / 256}")
MIN_LR=$(awk "BEGIN {printf \"%.8g\", ${EFFECTIVE_MAX_LR} * ${MIN_LR_RATIO}}")

echo "========================================="
echo " DINO ResNet50 Training"
echo "========================================="
echo " GPUs:           ${NUM_GPUS}"
echo " Data path:      ${DATA_PATH}"
echo " Output dir:     ${OUTPUT_DIR}"
echo " Batch/GPU:      ${BATCH_SIZE_PER_GPU}"
echo " Global crop:    ${GLOBAL_CROP_SIZE}, scale ${GLOBAL_CROPS_SCALE}"
echo " Local crop:     ${LOCAL_CROP_SIZE}, scale ${LOCAL_CROPS_SCALE}"
echo " Local crops #:  ${LOCAL_CROPS_NUMBER}"
echo " Base LR:        ${BASE_LR}"
echo " Effective LR:   ${EFFECTIVE_MAX_LR}"
echo " Min LR:         ${MIN_LR}"
echo "========================================="

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    main_dino.py \
    --arch resnet50 \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size_per_gpu ${BATCH_SIZE_PER_GPU} \
    --local_crops_number ${LOCAL_CROPS_NUMBER} \
    --global_crop_size ${GLOBAL_CROP_SIZE} \
    --local_crop_size ${LOCAL_CROP_SIZE} \
    --global_crops_scale ${GLOBAL_CROPS_SCALE} \
    --local_crops_scale ${LOCAL_CROPS_SCALE} \
    --optimizer lars \
    --lr ${BASE_LR} \
    --min_lr ${MIN_LR} \
    --weight_decay 1e-6 \
    --weight_decay_end 1e-6 \
    --out_dim 60000 \
    --norm_last_layer true \
    --use_bn_in_head true \
    --teacher_temp 0.07 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 50 \
    --freeze_last_layer 1 \
    --epochs 800 \
    --warmup_epochs 10 \
    --momentum_teacher 0.9995 \
    --clip_grad 0 \
    --use_fp16 false \
    --saveckp_freq 20 \
    --seed 0
