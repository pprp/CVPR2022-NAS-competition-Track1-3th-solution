#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J ppenv      # 作业名字
#SBATCH --gres=gpu:1   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load cuda/11.0
module load anaconda/2020.11 
module load nccl/2.9.6-1_cuda11.0
module load cudnn/8.1.1.33_CUDA11.0
source activate pp

JSON_PATH1=$1

# 为了优化速度问题，将数据集加载到内存中。
mkdir -p /dev/shm/imagenet-mini/val 
unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null

# 此处可填写运行程序的命令
python3 eval_supernet.py run \
  --backbone resnet48 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 2 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-autoslim2 \
  --log_freq 1 \
  --resume checkpoints/res48-autoslim2 \
  --json_path  $JSON_PATH1 \
  --image_dir /dev/shm/imagenet-mini 
