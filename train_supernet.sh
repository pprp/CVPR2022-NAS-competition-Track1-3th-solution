#!/bin/bash
source activate pp


# 将数据加载到内存中
# mkdir -p /dev/shm/imagenet-mini
# unzip -n /data/home/scv6681/run/data/imagenet_mini_train.zip -d /dev/shm/imagenet-mini > /dev/null
# unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null
IMAGE_DIR=/dev/shm/imagenet-mini


python train_supernet.py run \
  --backbone resnet48_mish \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 4 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/reproduct_rank_loss_flops_sanwich \
  --log_freq 50 \
  --visualdl_dir "./visualdl_log/reproduct_rank_loss_flops_sanwich" \
  --image_dir $IMAGE_DIR \

