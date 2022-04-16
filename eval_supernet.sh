#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J supnet      # 作业名字
#SBATCH --gres=gpu:8   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11 cuda/10.2 \
 cudnn/7.6.5.32_cuda10.2 nccl/2.9.6-1_cuda10.2
source activate pp

CUDA_VISIBLE_DEVICES=0,1 \
# 此处可填写运行程序的命令
python3 eval_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 1024 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 2 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-depth \
  --log_freq 1 \
  --resume checkpoints/res48-depth \
  --image_dir /data/public/imagenet2012

