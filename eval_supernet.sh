#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J supnet      # 作业名字
#SBATCH --gres=gpu:8   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11 cuda/10.2 \
 cudnn/7.6.5.32_cuda10.2 nccl/2.9.6-1_cuda10.2
source activate pp

# 此处可填写运行程序的命令
python3 eval_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 2048 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 2 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-depth \
  --log_freq 1 \
  --resume checkpoints/res48-depth \
  --image_dir /data/public/imagenet-mini

  # --image_dir /data/public/imagenet2012

# whole imagenet dataset 
# bs: 2048 forward: 101
# bs: 1024 forward: 67
# bs: 512  forward: 73

# imagenet-mini
# bs:1024 forward: 36.33 
# bs: 512 forward: 32.51
# bs: 256 forward: 30.20
# bs: 128 forward: 30.14

# single gpu imagenet-whole
# bs 2048 forward: 177s

# single gpu imagenet-mini
# bs 128 forward: 28s