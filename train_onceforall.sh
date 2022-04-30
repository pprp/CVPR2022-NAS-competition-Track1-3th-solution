#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J ofa      # 作业名字
#SBATCH --gres=gpu:8   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load cuda/11.0
module load anaconda/2020.11 
module load nccl/2.9.6-1_cuda11.0
module load cudnn/8.1.1.33_CUDA11.0
source activate pp

# 将数据加载到内存中
mkdir -p /dev/shm/imagenet2012
tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012 & tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012 
image_dir=/dev/shm/imagenet2012

# for fast running on imagenet-mini dataset 
# mkdir -p /dev/shm/imagenet-mini
# unzip -n /data/home/scv6681/run/data/imagenet_mini_val.zip -d /dev/shm/imagenet-mini > /dev/null
# image_dir=/data/public/imagenet-mini

##  info 
# ofa_run_2: 相比第一次增大了expand_ratio部分训练长度。
# ofa_run_3: 增大了第一个阶段的epoch个数。
# ofa_run_4: warmup 调整warmup，暂且没有处理,
#            expand warmup在第一个阶段为0，
#            expand阶段设置epoch个数边长(无用)，适当降低第一个阶段的epoch个数。
# ofa_run_5: 将expand部分warmup设置为0,并且每个阶段持续5个epoch
# ofa_run_6: 设置learning rate后期曲线变化明显, 减半后无效。

## exp
# 每个task的第一个phase要设置更多个epoch个数。
# warmup 在第一个phase设置为0
# expand ratio在

# 完整训练log: 131750

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
    max_epoch=20
    warmup=3
    dynamic_batch_size=2
    lr=0.001
  elif [ $phase -eq 2 ]; then
    max_epoch=28
    warmup=0
    dynamic_batch_size=2
    lr=0.001
  else
    max_epoch=34
    warmup=0
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
    max_epoch=45
    warmup=0
    dynamic_batch_size=4
    lr=0.0005 #0.001
  elif [ $phase -eq 2 ]; then
    max_epoch=50
    warmup=0
    dynamic_batch_size=4
    lr=0.0005 #0.001

  elif [ $phase -eq 3 ]; then
    max_epoch=55
    warmup=0
    dynamic_batch_size=4
    lr=0.0005 #0.001

  elif [ $phase -eq 4 ]; then
    max_epoch=60
    warmup=0
    dynamic_batch_size=4
    lr=0.0005 #0.001

  elif [ $phase -eq 5 ]; then
    max_epoch=65
    warmup=0
    dynamic_batch_size=4
    lr=0.0005 #0.001
  else
    max_epoch=70
    warmup=0
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