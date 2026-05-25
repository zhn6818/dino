#!/bin/bash

export OMP_NUM_THREADS=4

DATA_PATH="/data1/code/cv_workspace/datasets/jinxiang_linear"
BASE_OUTPUT="/data1/code/dino/output"
LINEAR_OUTPUT="${BASE_OUTPUT}/linear_eval"

COMMON_ARGS="--arch resnet50 \
    --data_path ${DATA_PATH} \
    --checkpoint_key teacher \
    --num_labels 4 \
    --batch_size_per_gpu 128 \
    --epochs 100 \
    --lr 0.001 \
    --num_workers 10 \
    --val_freq 1"

CHECKPOINTS=(
    "dino_resnet50_sgd_lr003_wd1e4_t0.02-0.05-50_m0.996|0"
    "dino_resnet50_sgd_lr003_wd1e4_t0.04-0.04-30_m0.996|1"
    "dino_resnet50_sgd_lr003_wd1e4_t0.04-0.07-30_m0.996|2"
    "dino_resnet50_sgd_lr003_wd1e4_t0.04-0.07-30_m0.9995|3"
    "dino_resnet50_sgd_lr003_wd1e4_t0.04-0.07-50_m0.996|4"
)

mkdir -p "${LINEAR_OUTPUT}"

PIDS=()

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r ckpt_dir gpu <<< "$entry"

    WEIGHTS="${BASE_OUTPUT}/${ckpt_dir}/checkpoint.pth"
    OUTPUT_DIR="${LINEAR_OUTPUT}/${ckpt_dir}"

    mkdir -p "${OUTPUT_DIR}"

    echo "============================================"
    echo "Linear eval: ${ckpt_dir} on GPU ${gpu}"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=${gpu} \
    torchrun --nproc_per_node=1 --master_port=$((29500 + gpu)) eval_linear.py \
        ${COMMON_ARGS} \
        --pretrained_weights "${WEIGHTS}" \
        --output_dir "${OUTPUT_DIR}" \
        2>&1 | tee "${OUTPUT_DIR}/eval_linear.log" &

    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} linear evaluations launched in parallel."
echo "PIDs: ${PIDS[*]}"

for pid in "${PIDS[@]}"; do
    wait ${pid}
    echo "Process ${pid} finished."
done

echo ""
echo "All linear evaluations completed."
echo ""
echo "=== Results Summary ==="
for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r ckpt_dir gpu <<< "$entry"
    LOG="${LINEAR_OUTPUT}/${ckpt_dir}/log.txt"
    if [ -f "${LOG}" ]; then
        BEST_ACC=$(grep "test_acc1" "${LOG}" | python3 -c "
import sys, json
best = 0
for line in sys.stdin:
    try:
        d = json.loads(line)
        acc = d.get('test_acc1', 0)
        if acc > best:
            best = acc
    except: pass
print(f'{best:.2f}')
" 2>/dev/null)
        echo "${ckpt_dir}: Top-1 Acc = ${BEST_ACC}%"
    else
        echo "${ckpt_dir}: log not found"
    fi
done
