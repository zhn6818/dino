#!/bin/bash

export OMP_NUM_THREADS=4

DATA_PATH="/data1/code/cv_workspace/datasets/jinxiang/"
BASE_OUTPUT="/data1/code/dino/output"

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

PARAM_GROUPS=(
    "0.04|0.07|30|0.996|0"
    "0.04|0.04|30|0.996|1"
    "0.04|0.07|30|0.9995|2"
    "0.02|0.05|50|0.996|3"
    "0.04|0.07|50|0.996|4"
)

PIDS=()

for params in "${PARAM_GROUPS[@]}"; do
    IFS='|' read -r warmup_t t t_epochs momentum gpu <<< "$params"

    DIR_NAME="dino_resnet50_sgd_lr003_wd1e4_t${warmup_t}-${t}-${t_epochs}_m${momentum}"
    OUTPUT_DIR="${BASE_OUTPUT}/${DIR_NAME}"

    mkdir -p "${OUTPUT_DIR}"

    echo "============================================"
    echo "Training: ${DIR_NAME} on GPU ${gpu}"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=${gpu} \
    torchrun --nproc_per_node=1 --master_port=$((29500 + gpu)) main_dino.py \
        ${COMMON_ARGS} \
        --output_dir "${OUTPUT_DIR}" \
        --warmup_teacher_temp ${warmup_t} \
        --teacher_temp ${t} \
        --warmup_teacher_temp_epochs ${t_epochs} \
        --momentum_teacher ${momentum} \
        2>&1 | tee "${OUTPUT_DIR}/train.log" &

    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} experiments launched in parallel."
echo "PIDs: ${PIDS[*]}"

for pid in "${PIDS[@]}"; do
    wait ${pid}
    echo "Process ${pid} finished."
done

echo "All experiments completed."
