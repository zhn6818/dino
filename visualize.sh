#!/bin/bash
# DINO Attention 可视化脚本
#
# 用法：
#   bash visualize.sh                  # 默认生成热力图+mask
#   bash visualize.sh docker           # 容器内执行
#   IMAGE=img/test.jpg bash visualize.sh  # 指定图片

MODE=${1:-local}
IMAGE=${IMAGE:-img/test4.jpg}
CHECKPOINT=${CHECKPOINT:-dino_output/checkpoint.pth}
OUTPUT_DIR=${OUTPUT_DIR:-attention_maps}

VIS_CMD="source /opt/miniconda3/etc/profile.d/conda.sh && conda activate ai && \
    python visualize_attention.py \
        --arch vit_small \
        --patch_size 16 \
        --pretrained_weights ${CHECKPOINT} \
        --checkpoint_key teacher \
        --image_path ${IMAGE} \
        --output_dir ${OUTPUT_DIR} \
        --threshold 0.6"

case "$MODE" in
    docker)
        docker exec \
            -w /data2/zhn/code/dino \
            JHCVTrain \
            bash -c "$VIS_CMD"
        ;;
    local)
        eval "$VIS_CMD"
        ;;
    *)
        echo "用法: bash visualize.sh [docker|local]"
        echo "  docker - 在 JHCVTrain 容器中执行"
        echo "  local  - 在当前环境直接执行"
        exit 1
        ;;
esac
