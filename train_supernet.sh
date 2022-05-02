#!/bin/bash

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11 
module load cuda/11.1
module load nccl/2.11.4-1_cuda11.1
module load cudnn/8.2.1_cuda11.x 
source activate pp

# 将数据加载到内存中
mkdir -p /dev/shm/imagenet2012
tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012 & tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012 



# 此处可填写运行程序的命令
# fairnas0: 执行autoslim+fairnas方法
# fairnas1: 删除autoslim，仅执行fairnas方法，影响不大
# fairnas2: autoslim + random 8 times 进行对比
# fairnas3: 尝试每个arch多进行几次迭代
# fairnas4: 在3基础上尝试全量训练。

python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 4 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-fairnas \
  --log_freq 10 \
  --image_dir /dev/shm/imagenet2012
  # /data/public/imagenet-mini
  # /dev/shm/imagenet2012 \
  # --resume checkpoints/res48-fairnas2 \


