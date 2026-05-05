#!/bin/bash
# DINO 训练脚本
#
# 用法：
#   bash train.sh [选项]
#
# 训练样例：
#
#   # ViT-Small patch16（默认轻量配置）
#   bash train.sh --arch vit_small --patch_size 16
#
#   # ViT-Small patch8（更精细，显存更大）
#   bash train.sh --arch vit_small --patch_size 8 --batch_size_per_gpu 10
#
#   # ViT-Base patch16（更大模型，需更多显存）
#   bash train.sh --arch vit_base --patch_size 16 --batch_size_per_gpu 8
#
#   # ViT-Base patch8（最强效果，显存需求最大）
#   bash train.sh --arch vit_base --patch_size 8 --batch_size_per_gpu 4
#
#   # ResNet-50（卷积网络，无需 patch_size）
#   bash train.sh --arch resnet50 --mode docker
#
#   # 容器内执行
#   bash train.sh --mode docker --arch vit_small --patch_size 16
#
#   # 后台运行
#   nohup bash train.sh --arch vit_small --patch_size 16 --mode docker > train_vit_small_p16.log 2>&1 &
#   nohup bash train.sh --arch resnet50 --mode docker > train_resnet50.log 2>&1 &
#   nohup bash train.sh --arch xcit_small_12_p16 --mode docker > train_xcit_small_12_p16.log 2>&1 &
#   bash train.sh --mode docker --arch resnet50
# ── 默认参数 ──
MODE=local
ARCH=vit_small
PATCH_SIZE=
DATA_PATH=/data2/zhn/code/data/jinxiang/
EPOCHS=1000
BATCH_SIZE_PER_GPU=2
LOCAL_CROPS_NUMBER=32
GLOBAL_CROP_SIZE=512
LOCAL_CROP_SIZE=256
GLOBAL_CROPS_SCALE="0.05 0.2"
LOCAL_CROPS_SCALE="0.01 0.08"
USE_FP16=true
OPTIMIZER=adamw
LR=0.0005
WARMUP_EPOCHS=10
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.4
SAVECKP_FREQ=20
NUM_WORKERS=4
SEED=0
MASTER_PORT=29503

# ── 解析参数 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)                MODE="$2";                shift 2 ;;
        --arch)                ARCH="$2";                shift 2 ;;
        --patch_size)          PATCH_SIZE="$2";          shift 2 ;;
        --data_path)           DATA_PATH="$2";           shift 2 ;;
        --epochs)              EPOCHS="$2";              shift 2 ;;
        --batch_size_per_gpu)  BATCH_SIZE_PER_GPU="$2";  shift 2 ;;
        --local_crops_number)  LOCAL_CROPS_NUMBER="$2";  shift 2 ;;
        --global_crop_size)    GLOBAL_CROP_SIZE="$2";    shift 2 ;;
        --local_crop_size)     LOCAL_CROP_SIZE="$2";     shift 2 ;;
        --global_crops_scale)  GLOBAL_CROPS_SCALE="$2";  shift 2 ;;
        --local_crops_scale)   LOCAL_CROPS_SCALE="$2";   shift 2 ;;
        --use_fp16)            USE_FP16="$2";            shift 2 ;;
        --optimizer)           OPTIMIZER="$2";           shift 2 ;;
        --lr)                  LR="$2";                  shift 2 ;;
        --warmup_epochs)       WARMUP_EPOCHS="$2";       shift 2 ;;
        --weight_decay)        WEIGHT_DECAY="$2";        shift 2 ;;
        --weight_decay_end)    WEIGHT_DECAY_END="$2";    shift 2 ;;
        --saveckp_freq)        SAVECKP_FREQ="$2";        shift 2 ;;
        --num_workers)         NUM_WORKERS="$2";         shift 2 ;;
        --seed)                SEED="$2";                shift 2 ;;
        --master_port)         MASTER_PORT="$2";         shift 2 ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ── GPU 检测 ──
TOTAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$TOTAL_GPUS" -le 1 ]; then
    GPUS="0"
    NUM_GPUS=1
else
    GPUS=$(seq -s, 1 $((TOTAL_GPUS - 1)))
    NUM_GPUS=$((TOTAL_GPUS - 1))
fi
echo "检测到 ${TOTAL_GPUS} 张 GPU，使用 GPU ${GPUS}（${NUM_GPUS} 张）进行训练"

# ── 输出目录：根据架构和参数自动命名 ──
if [ -n "$PATCH_SIZE" ]; then
    OUTPUT_DIR="./dino_output/${ARCH}_p${PATCH_SIZE}"
    PATCH_ARG="--patch_size ${PATCH_SIZE}"
else
    OUTPUT_DIR="./dino_output/${ARCH}"
    PATCH_ARG=""
fi
# 非默认裁剪尺寸时追加到目录名
if [ "$GLOBAL_CROP_SIZE" != "224" ] || [ "$LOCAL_CROP_SIZE" != "96" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_gc${GLOBAL_CROP_SIZE}_lc${LOCAL_CROP_SIZE}"
fi
echo "输出目录: ${OUTPUT_DIR}"

TRAIN_CMD="source /opt/miniconda3/etc/profile.d/conda.sh && conda activate ai && \
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        main_dino.py \
        --arch ${ARCH} \
        ${PATCH_ARG} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --epochs ${EPOCHS} \
        --batch_size_per_gpu ${BATCH_SIZE_PER_GPU} \
        --local_crops_number ${LOCAL_CROPS_NUMBER} \
        --global_crop_size ${GLOBAL_CROP_SIZE} \
        --local_crop_size ${LOCAL_CROP_SIZE} \
        --global_crops_scale ${GLOBAL_CROPS_SCALE} \
        --local_crops_scale ${LOCAL_CROPS_SCALE} \
        --use_fp16 ${USE_FP16} \
        --optimizer ${OPTIMIZER} \
        --lr ${LR} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --weight_decay ${WEIGHT_DECAY} \
        --weight_decay_end ${WEIGHT_DECAY_END} \
        --saveckp_freq ${SAVECKP_FREQ} \
        --num_workers ${NUM_WORKERS} \
        --seed ${SEED}"

# ── Docker 命令前缀（无权限时自动加 sudo） ──
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
fi

case "$MODE" in
    docker)
        $DOCKER exec \
            -w /data2/zhn/code/dino \
            -e CUDA_VISIBLE_DEVICES=${GPUS} \
            JHCVTrain \
            bash -c "$TRAIN_CMD"
        ;;
    local)
        export CUDA_VISIBLE_DEVICES=${GPUS}
        eval "$TRAIN_CMD"
        ;;
    *)
        echo "未知模式: ${MODE}，请使用 docker 或 local"
        exit 1
        ;;
esac
