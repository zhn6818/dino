#!/bin/bash
# DINO 可视化脚本
#
# 用法：
#   bash visualize.sh                            # 默认 ViT 注意力图
#   bash visualize.sh resnet                     # ResNet50 Grad-CAM
#   bash visualize.sh docker                     # 容器内 ViT
#   bash visualize.sh docker resnet              # 容器内 ResNet50
#   IMAGE=img/test.jpg bash visualize.sh resnet  # 指定图片

MODE=local
ARCH=vit

for arg in "$@"; do
    case "$arg" in
        docker) MODE=docker ;;
        resnet) ARCH=resnet ;;
        vit)    ARCH=vit ;;
    esac
done

IMAGE=${IMAGE:-img/test4.jpg}
if [ "$ARCH" = "resnet" ]; then
    CHECKPOINT=${CHECKPOINT:-dino_output/resnet50_gc512_lc256/checkpoint.pth}
else
    CHECKPOINT=${CHECKPOINT:-dino_output/checkpoint.pth}
fi
OUTPUT_DIR=${OUTPUT_DIR:-attention_maps}

if [ "$ARCH" = "resnet" ]; then
    OUTPUT_DIR=${OUTPUT_DIR/attention_maps/gradcam_maps}
    VIS_CMD="source /opt/miniconda3/etc/profile.d/conda.sh && conda activate ai && \
        python visualize_resnet.py \
            --arch resnet50 \
            --pretrained_weights ${CHECKPOINT} \
            --checkpoint_key teacher \
            --image_path ${IMAGE} \
            --output_dir ${OUTPUT_DIR}"
else
    VIS_CMD="source /opt/miniconda3/etc/profile.d/conda.sh && conda activate ai && \
        python visualize_attention.py \
            --arch vit_small \
            --patch_size 16 \
            --pretrained_weights ${CHECKPOINT} \
            --checkpoint_key teacher \
            --image_path ${IMAGE} \
            --output_dir ${OUTPUT_DIR} \
            --threshold 0.6"
fi

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
esac
