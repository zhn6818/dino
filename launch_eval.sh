#!/bin/bash
export PATH=/opt/miniconda3/envs/ai/bin:$PATH
export OMP_NUM_THREADS=4
cd /data1/code/dino
bash eval_linear.sh
