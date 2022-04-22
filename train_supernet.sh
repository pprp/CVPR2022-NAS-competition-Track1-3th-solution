#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J supnet      # 作业名字
#SBATCH --gres=gpu:8   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11 cuda/10.2 \
 cudnn/7.6.5.32_cuda10.2 nccl/2.9.6-1_cuda10.2
source activate pp

# 将数据加载到内存中
mkdir -p /dev/shm/imagenet2012
tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012 & tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012 

# 此处可填写运行程序的命令

python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 4 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-ofa \
  --log_freq 10 \
  --image_dir /dev/shm/imagenet2012 \
  --resume checkpoints/res48-ofa \


