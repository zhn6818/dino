#!/bin/bash
# DINO 全层 Attention + PCA 可视化脚本
#
# 用法：
#   bash visualize_all.sh                  # 默认执行
#   bash visualize_all.sh docker           # 容器内执行
#   IMAGE=img/test.jpg bash visualize_all.sh  # 指定图片

MODE=${1:-local}
IMAGE=${IMAGE:-img/test4.jpg}
CHECKPOINT=${CHECKPOINT:-dino_output/checkpoint.pth}
OUTPUT_DIR=${OUTPUT_DIR:-attention_maps_all}

# 按图片文件名创建子文件夹，避免多次测试覆盖
IMG_NAME=$(basename "${IMAGE}" | sed 's/\.[^.]*$//')
FULL_OUTPUT_DIR="${OUTPUT_DIR}/${IMG_NAME}"

VIS_CMD="source /opt/miniconda3/etc/profile.d/conda.sh && conda activate ai && \
    python visualize_all_layers.py \
        --arch vit_small \
        --patch_size 16 \
        --pretrained_weights ${CHECKPOINT} \
        --checkpoint_key teacher \
        --image_path ${IMAGE} \
        --output_dir ${FULL_OUTPUT_DIR} \
        --threshold 0.6"

# Docker 命令前缀（无权限时自动加 sudo）
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
fi

case "$MODE" in
    docker)
        $DOCKER exec \
            -w /data2/zhn/code/dino \
            JHCVTrain \
            bash -c "$VIS_CMD"
        ;;
    local)
        eval "$VIS_CMD"
        ;;
    *)
        echo "用法: bash visualize_all.sh [docker|local]"
        echo "  docker - 在 JHCVTrain 容器中执行"
        echo "  local  - 在当前环境直接执行"
        exit 1
        ;;
esac
