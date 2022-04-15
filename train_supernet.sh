#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 64 \
  --lr 0.00025 \
  --warmup 5 \
  --dyna_batch_size 2 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-depth \
  --log_freq 1 \
  --resume checkpoints/res48-depth \
  --image_dir /media/niu/niu_g/data/imagenet > supernet_log 2>&1 &
  #  --image_dir /home/xlz/data/imagenet


