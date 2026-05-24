#!/bin/bash

DATA_PATH="/root/autodl-tmp/data/jinxiang/jinxiang/"
BASE_OUTPUT="/root/autodl-tmp/data/dinomodel/"
COMMON_ARGS="--arch resnet50 \
    --data_path ${DATA_PATH} \
    --batch_size_per_gpu 32 \
    --local_crops_number 8 \
    --epochs 100 \
    --lr 0.0005 \
    --warmup_epochs 10 \
    --use_fp16 true \
    --optimizer lars \
    --saveckp_freq 20 \
    --num_workers 4"

# Define parameter groups: each row is one experiment
# Format: wd wd_end warmup_t t t_epochs momentum
PARAM_GROUPS=(
    "0.04|0.4|0.04|0.07|30|0.996"
    "0.04|0.4|0.04|0.04|30|0.996"
    "0.04|0.4|0.04|0.07|30|0.9995"
    "0.01|0.1|0.04|0.07|30|0.996"
    "0.04|0.4|0.02|0.05|50|0.996"
)

for params in "${PARAM_GROUPS[@]}"; do
    IFS='|' read -r wd wd_end warmup_t t t_epochs momentum <<< "$params"

    # Build output dir name from parameters
    DIR_NAME="dino_resnet50_wd${wd}-${wd_end}_t${warmup_t}-${t}-${t_epochs}_m${momentum}"
    OUTPUT_DIR="${BASE_OUTPUT}/${DIR_NAME}"

    mkdir -p "${OUTPUT_DIR}"

    echo "============================================"
    echo "Training: ${DIR_NAME}"
    echo "============================================"

    torchrun --nproc_per_node=1 --master_port=29500 main_dino.py \
        ${COMMON_ARGS} \
        --output_dir "${OUTPUT_DIR}" \
        --weight_decay ${wd} \
        --weight_decay_end ${wd_end} \
        --warmup_teacher_temp ${warmup_t} \
        --teacher_temp ${t} \
        --warmup_teacher_temp_epochs ${t_epochs} \
        --momentum_teacher ${momentum} \
        2>&1 | tee "${OUTPUT_DIR}/train.log"

    echo "Finished: ${DIR_NAME}"
    echo ""
done

echo "All experiments completed."
