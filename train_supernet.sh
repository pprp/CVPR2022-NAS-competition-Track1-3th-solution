#!/bin/bash
#SBATCH -J t32th      # 作业名字

module load anaconda/2020.11
module load cuda/11.0
module load cudnn/8.1.0.77_CUDA11.1
module load nccl/2.9.6-1_cuda11.0
source activate pp22

mkdir -p /dev/shm/imagenet-mini
cp -r /data/public/imagenet-mini/ /dev/shm

image_dir=/dev/shm/imagenet-mini
# 此处可填写运行程序的命令

# ignore warning 
python train_supernet.py run \
  --backbone resnet48_prelu \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 4 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/32th_prelu_flops_lam1_sandwich_balance_2times \
  --log_freq 50 \
  --visualdl_dir "./visualdl_log/32th_prelu_flops_lam1_sandwich_balance_2times" \
  --image_dir $image_dir \


# rankloss_run7: 使用rank_loss, coeff=1, prelu, sample 2 times zencore去掉shortcut [81.10]
# rankloss_cosine: 8 gpu + cosine lambda scheduler(max=10), prelu [failed to converge]
# rankloss_cosine_run2: max=5, prelu, zenscore, 

# rkloss_mish_zenscore_warmup15_max15_3times
# 16th_rkloss_mish_zenscore_lamb1_3times_dis33

######################################## balance nas 

# 17th_rkloss_mish_flops_wm20_max2_balance 

# 25th_rkloss_mish_flops_4stages_max_15_sandwich_2times: 

# 22th_reproduction_mish_flops_warmup20_max2_sandwich_2times:

# 32th_prelu_flops_lam1_sandwich_balance_2times: 