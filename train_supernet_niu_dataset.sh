#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J t19th      # 作业名字
# 需要使用的卡数

module load cuda/11.0
module load anaconda/2020.11
module load nccl/2.9.6-1_cuda11.0
module load cudnn/8.1.1.33_CUDA11.0
source activate pp


# python -m pip install paddlepaddle-gpu==2.0.2.post110 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

# 将数据加载到内存中
mkdir -p /dev/shm/imagenet-mini
# unzip -n /data/home/scv6681/run/data/imagenet_mini_train.zip -d /dev/shm/imagenet-mini > /dev/null
# unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null

cp -r /data/home/scv6681/run/data/imagenet-mini-specific-class-expand/* /dev/shm/imagenet-mini

image_dir=/dev/shm/imagenet-mini
# 此处可填写运行程序的命令

# ignore warning 
python train_supernet.py run \
  --backbone resnet48_mish \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 4 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/23th_niudataset_mish_flops_warmup20_max2_sandwich_2times \
  --log_freq 50 \
  --visualdl_dir "./visualdl_log/23th_niudataset_mish_flops_warmup20_max2_sandwich_2times" \
  --image_dir $image_dir \


# 按照牛老师说的方案，统计每个类别的准确率，找到0.3以下的类别进行训练集扩充。
# 23th_niudataset_mish_flops_warmup20_max2_sandwich_2times