#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J ofa      # 作业名字
#SBATCH --gres=gpu:4   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load cuda/11.0
module load anaconda/2020.11 
module load nccl/2.9.6-1_cuda11.0
module load cudnn/8.1.1.33_CUDA11.0
source activate pp

# 将数据加载到内存中
# mkdir -p /dev/shm/imagenet2012
# tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012 & tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012 

# image_dir=/dev/shm/imagenet2012

# for fast running on imagenet-mini dataset 
mkdir -p /dev/shm/imagenet-mini
unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null

image_dir=/data/public/imagenet-mini


# 此处可填写运行程序的命令

depth_phase_num=3
for ((phase=1; phase<=$depth_phase_num; phase++))
do
  save_dir=checkpoints/res48_ofa/res48_ofa_depth_$phase

  if [ $phase -eq 1 ]; then
    resume=None
  else
    resume=checkpoints/res48_ofa/res48_ofa_depth_$(($phase-1))
  fi

  if [ $phase -eq 1 ]; then
    max_epoch=2 #10
    warmup=1
    dynamic_batch_size=2
    lr=0.001
  elif [ $phase -eq 2 ]; then
    max_epoch=4 #18
    warmup=1
    dynamic_batch_size=2
    lr=0.001
  else
    max_epoch=6 #24
    warmup=1
    dynamic_batch_size=2
    lr=0.001
  fi 

  python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch $max_epoch \
  --batch_size 256 \
  --lr $lr \
  --warmup $warmup \
  --dyna_batch_size $dynamic_batch_size \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir $save_dir \
  --log_freq 10 \
  --image_dir $image_dir \
  --task depth \
  --phase $phase \
  --resume $resume
done


width_phase_num=6
for ((phase=1; phase<=width_phase_num; phase++))
do
  save_dir=checkpoints/res48_ofa/res48_ofa_width_$phase

  if [ $phase -eq 1 ]; then 
    resume=checkpoints/res48_ofa/res48_ofa_depth_$depth_phase_num
  else
    resume=checkpoints/res48_ofa/res48_ofa_width_$(($phase-1))
  fi

  if [ $phase -eq 1 ]; then
    max_epoch=8 #29
    warmup=1
    dynamic_batch_size=4
    lr=0.0005 #0.001
  elif [ $phase -eq 2 ]; then
    max_epoch=10 #34
    warmup=1
    dynamic_batch_size=4
    lr=0.0005 #0.001

  elif [ $phase -eq 3 ]; then
    max_epoch=12 #39
    warmup=1
    dynamic_batch_size=4
    lr=0.0005 #0.001

  elif [ $phase -eq 4 ]; then
    max_epoch=14 #44
    warmup=1
    dynamic_batch_size=4
    lr=0.0005 #0.001

  elif [ $phase -eq 5 ]; then
    max_epoch=16 #49
    warmup=1
    dynamic_batch_size=4
    lr=0.0005 #0.001
  else
    max_epoch=18 #54
    warmup=1
    dynamic_batch_size=4
    lr=0.0005 #0.001
  fi 


  python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch $max_epoch \
  --batch_size 256 \
  --lr $lr \
  --warmup $warmup \
  --dyna_batch_size $dyna_batch_size \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir $save_dir \
  --log_freq 10 \
  --image_dir $image_dir \
  --task expand_ratio \
  --phase $phase \
  --resume $resume
done