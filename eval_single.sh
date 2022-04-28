#!/bin/bash
# eval single network [FOR DEBUG]

module load cuda/11.0
module load anaconda/2020.11 
module load nccl/2.9.6-1_cuda11.0
module load cudnn/8.1.1.33_CUDA11.0
source activate pp

python -u eval_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 128 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 2 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-depth \
  --log_freq 1 \
  --resume checkpoints/res48-depth \
  --json_path  checkpoints/jsons/SubTest_8_5.json \
  --image_dir /data/public/imagenet-mini 