#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J autoslim      # 作业名字
#SBATCH --gres=gpu:8   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load cuda/11.0
module load anaconda/2020.11 
module load nccl/2.9.6-1_cuda11.0
module load cudnn/8.1.1.33_CUDA11.0
source activate pp

# python -m pip install paddlepaddle-gpu==2.0.2.post110 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

# 将数据加载到内存中
# mkdir -p /dev/shm/imagenet2012
# tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012 & tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012 

IMAGEDIR=/data/public/imagenet-mini
# 此处可填写运行程序的命令
python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --task expand_ratio \
  --dyna_batch_size 4 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/test_del \
  --log_freq 50 \
  --image_dir $IMAGEDIR  \
  --phase 1 
  # --resume checkpoints/res48-autoslim \
