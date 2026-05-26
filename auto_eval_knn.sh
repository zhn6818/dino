#!/bin/bash
export PATH=/opt/miniconda3/envs/ai/bin:$PATH

# ============ 可配置参数 ============
GPUS="0"                                                # k-NN 评估使用的 GPU
ARCH="resnet50"                                         # 模型架构
CHECKPOINT_KEY="teacher"                                # checkpoint 中加载的 key
CROP_SIZE=448                                           # 评估裁剪尺寸: 224 或 448
CKPT_DIR="/data1/code/dino/output2/dino_resnet50_sgd_lr003_wd1e4_t0.04-0.07-50_m0.996"
DATA_PATH="/data1/code/cv_workspace/datasets/jinxiang_linear"
MASTER_PORT=29600                                       # torchrun 端口，避免与训练端口冲突
CHECK_INTERVAL=300                                      # 检查间隔（秒）
# ===================================

KNN_OUTPUT="${CKPT_DIR}/knn_eval"
PROCESSED_LIST="${KNN_OUTPUT}/.processed"

mkdir -p "${KNN_OUTPUT}"
touch "${PROCESSED_LIST}"

echo "=== k-NN Auto Eval Started ==="
echo "Monitoring: ${CKPT_DIR}"
echo "Eval data: ${DATA_PATH}"
echo "GPU: ${GPUS}"
echo ""

eval_checkpoint() {
    local ckpt_path=$1
    local ckpt_name=$(basename "$ckpt_path")

    grep -q "^${ckpt_name}$" "${PROCESSED_LIST}" 2>/dev/null && return 0

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] New checkpoint: ${ckpt_name}"

    CUDA_VISIBLE_DEVICES=${GPUS} \
    torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} eval_knn.py \
        --arch ${ARCH} \
        --crop_size ${CROP_SIZE} \
        --checkpoint_key ${CHECKPOINT_KEY} \
        --pretrained_weights "$ckpt_path" \
        --data_path "${DATA_PATH}" \
        --num_workers 8 \
        2>&1 | tee "${KNN_OUTPUT}/knn_${ckpt_name}.log"

    if [ $? -eq 0 ]; then
        echo "${ckpt_name}" >> "${PROCESSED_LIST}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${ckpt_name}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed: ${ckpt_name}, will retry next cycle"
        # Remove failed log so it can retry
        rm -f "${KNN_OUTPUT}/knn_${ckpt_name}.log"
    fi
    echo ""
}

while true; do
    # Evaluate all unprocessed checkpoints
    for ckpt in ${CKPT_DIR}/checkpoint*.pth; do
        [ -f "$ckpt" ] || continue
        eval_checkpoint "$ckpt"
    done

    # Check if training is still running
    TRAINING_RUNNING=$(ps aux | grep main_dino.py | grep -v grep | wc -l)

    if [ "$TRAINING_RUNNING" -eq 0 ]; then
        # Final pass: re-evaluate checkpoint.pth (always the latest)
        sed -i '/^checkpoint\.pth$/d' "${PROCESSED_LIST}"
        eval_checkpoint "${CKPT_DIR}/checkpoint.pth"

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
    result=$(grep "10-NN" "$log" | tail -1)
    echo "${name}: ${result}"
done
