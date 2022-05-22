#!/bin/bash

JSON_PATH1=$1
RESUME_PATH="checkpoints/reproduct_rank_loss_flops_sanwich"

# 为了优化速度问题，将数据集加载到内存中。
# mkdir -p /dev/shm/imagenet-mini
# unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null
IMAGE_DIR=/dev/shm/imagenet-mini

python3 -u eval_supernet.py run \
  --backbone resnet48_mish \
  --batch_size 64 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 2 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir $RESUME_PATH \
  --log_freq 1 \
  --resume $RESUME_PATH \
  --json_path  $JSON_PATH1 \
  --image_dir  $IMAGE_DIR
