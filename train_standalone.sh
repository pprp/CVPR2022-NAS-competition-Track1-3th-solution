#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J standalone      # 作业名字
#SBATCH --gres=gpu:4   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11 cuda/11.1 cudnn/8.2.1_cuda11.x nccl/2.11.4-1_cuda11.1
source activate pp

# 为了优化速度问题，将数据集加载到内存中。
#mkdir -p /dev/shm/imagenet-mini/val
#unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini
mkdir -p /dev/shm/imagenet2012
tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012
tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012


image_dir=/dev/shm/imagenet2012

arch1=$1
arch2=$2
arch3=$3

log_name1=log/$arch1.log
log_name2=log/$arch2.log
log_name3=log/$arch3.log

python3 -u -m paddle.distributed.launch --gpu 0,1,2,3 pretrain.py run \
        --arch $arch1 \
        --image_dir $image_dir \
        --batch_size 1024 \
        --max_epoch 90 \
        --warmup 1 > $log_name1

python3 -u -m paddle.distributed.launch --gpu 0,1,2,3 pretrain.py run \
        --arch $arch2 \
        --image_dir $image_dir \
        --batch_size 1024 \
        --max_epoch 90 \
        --warmup 1 > $log_name2

python3 -u -m paddle.distributed.launch --gpu 0,1,2,3 pretrain.py run \
        --arch $arch3 \
        --image_dir $image_dir \
        --batch_size 1024 \
        --max_epoch 90 \
        --warmup 1 > $log_name3

