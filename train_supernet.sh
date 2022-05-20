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
unzip -n /data/home/scv6681/run/data/imagenet_mini_train.zip -d /dev/shm/imagenet-mini > /dev/null
unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null

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
  --save_dir checkpoints/19th_rkloss_mish_flops_latedecay_sandwich_2times \
  --log_freq 50 \
  --visualdl_dir "./visualdl_log/19th_rkloss_mish_flops_latedecay_sandwich_2times" \
  --image_dir $image_dir \


# rankloss_run7: 使用rank_loss, coeff=1, prelu, sample 2 times zencore去掉shortcut [81.10]
# rankloss_cosine: 8 gpu + cosine lambda scheduler(max=10), prelu [failed to converge]
# rankloss_cosine_run2: max=5, prelu, zenscore, 

# rkloss_mish_zenscore_warmup15_max15_3times
# rkloss_mish_zenscore_warmup15_max15_3times 
# 19th_rkloss_mish_flops_latedecay_sandwich_2times

# 备注：rkloss_mish_zenscore_warmup15_max15_3times results文件夹中，后半段是rkloss_mish_zenscore_warmup15_max15_3times