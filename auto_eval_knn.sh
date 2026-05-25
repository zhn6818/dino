#!/bin/bash
export PATH=/opt/miniconda3/envs/ai/bin:$PATH

# ============ 可配置参数 ============
GPUS="0"                                                # k-NN 评估使用的 GPU
ARCH="resnet50"                                         # 模型架构
CHECKPOINT_KEY="teacher"                                # checkpoint 中加载的 key
CKPT_DIR="/data1/code/dino/output2/dino_resnet50_sgd_lr003_wd1e4_t0.04-0.07-50_m0.996"
DATA_PATH="/data1/code/cv_workspace/datasets/jinxiang_linear"
KNN_OUTPUT="${CKPT_DIR}/knn_eval"
MASTER_PORT=29600
CHECK_INTERVAL=300                                      # 检查间隔（秒）
# ===================================

PROCESSED_LIST="${KNN_OUTPUT}/.knn_processed"

mkdir -p "${KNN_OUTPUT}"
touch "${PROCESSED_LIST}"

echo "=== k-NN Auto Eval Started ==="
echo "Monitoring: ${CKPT_DIR}"
echo "Eval data: ${DATA_PATH}"
echo "GPU: ${GPUS}"
echo ""

while true; do
    TRAINING_RUNNING=$(ps aux | grep main_dino.py | grep -v grep | wc -l)

    for ckpt in ${CKPT_DIR}/checkpoint*.pth; do
        [ -f "$ckpt" ] || continue
        ckpt_name=$(basename "$ckpt")

        grep -q "^${ckpt_name}$" "${PROCESSED_LIST}" 2>/dev/null && continue

        echo "============================================"
        echo "New checkpoint: ${ckpt_name}"
        echo "Running k-NN evaluation..."
        echo "============================================"

        CUDA_VISIBLE_DEVICES=${GPUS} \
        torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} eval_knn.py \
            --arch ${ARCH} \
            --checkpoint_key ${CHECKPOINT_KEY} \
            --pretrained_weights "$ckpt" \
            --data_path "${DATA_PATH}" \
            --num_workers 8 \
            2>&1 | tee "${KNN_OUTPUT}/knn_${ckpt_name}.log"

        echo "${ckpt_name}" >> "${PROCESSED_LIST}"
        echo "Finished: ${ckpt_name}"
        echo ""
    done

    if [ "$TRAINING_RUNNING" -eq 0 ]; then
        if [ -f "${CKPT_DIR}/checkpoint.pth" ]; then
            grep -q "^checkpoint.pth$" "${PROCESSED_LIST}" 2>/dev/null || {
                echo "Processing final checkpoint.pth..."
                CUDA_VISIBLE_DEVICES=${GPUS} \
                torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} eval_knn.py \
                    --arch ${ARCH} \
                    --checkpoint_key ${CHECKPOINT_KEY} \
                    --pretrained_weights "${CKPT_DIR}/checkpoint.pth" \
                    --data_path "${DATA_PATH}" \
                    --num_workers 8 \
                    2>&1 | tee "${KNN_OUTPUT}/knn_checkpoint.pth.log"
                echo "checkpoint.pth" >> "${PROCESSED_LIST}"
            }
        fi
        echo "Training finished. All checkpoints evaluated."
        break
    fi

    sleep ${CHECK_INTERVAL}
done

echo ""
echo "=== k-NN Results Summary ==="
for log in ${KNN_OUTPUT}/knn_*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log)
    result=$(grep -E "Top-1|Acc" "$log" | tail -1)
    echo "${name}: ${result}"
done
