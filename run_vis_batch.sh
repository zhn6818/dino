#!/bin/bash
CKPT_DIR="dino_output/resnet50"
for ckpt in ${CKPT_DIR}/checkpoint[0-9][0-9][0-9][0-9].pth; do
    name=$(basename "$ckpt" .pth)
    epoch=${name#checkpoint}
    outdir="gradcam_maps/${epoch}"
    echo "=== Epoch ${epoch} ==="
    python visualize_resnet.py --pretrained_weights "$ckpt" --output_dir "$outdir" --image_path "img/test5.jpg"
done
