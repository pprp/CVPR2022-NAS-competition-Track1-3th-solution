#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J e32th      # 作业名字
#SBATCH --gres=gpu:1   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11
module load cuda/11.1
module load cudnn/8.1.0.77_CUDA11.1
module load nccl/2.9.6-1_cuda11.0
source activate pp22

JSON_PATH1=$1

# 为了优化速度问题，将数据集加载到内存中。
# mkdir -p /dev/shm/imagenet-mini/val 
# unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null

# mkdir -p /dev/shm/imagenet-mini
## unzip -n /data/home/scv6681/run/data/imagenet_mini_train.zip -d /dev/shm/imagenet-mini > /dev/null
# unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null
# image_dir=/dev/shm/imagenet-mini

# double imagenet-mini
#mkdir -p /dev/shm/imagenet-mini-double
#cp -rf /data/home/scv6681/run/data/imagenet-mini-double/val  /dev/shm/imagenet-mini-double
#image_dir=/dev/shm/imagenet-mini-double

mkdir -p /dev/shm/imagenet-mini
# unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null
cp -r /data/public/imagenet-mini/val /dev/shm/imagenet-mini
image_dir=/dev/shm/imagenet-mini

# 此处可填写运行程序的命令
python3 -u eval_supernet.py run \
  --backbone resnet48_prelu \
  --batch_size 128 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 2 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/32th_prelu_flops_lam1_sandwich_balance_2times \
  --log_freq 1 \
  --resume checkpoints/32th_prelu_flops_lam1_sandwich_balance_2times \
  --json_path  $JSON_PATH1 \
  --image_dir  $image_dir
