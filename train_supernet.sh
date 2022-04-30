#!/bin/bash
<<<<<<< HEAD
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J flpsd      # 作业名字
=======
>>>>>>> 6b096393910c6f5c0e97472b23d02d117817f3a4

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11 cuda/11.1 cudnn/8.2.1_cuda11.x nccl/2.11.4-1_cuda11.1
source activate pp

# 将数据加载到内存中
#mkdir -p /dev/shm/imagenet2012
#tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012
#tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012
#
#image_dir=/dev/shm/imagenet2012
image_dir=/data/public/imagenet-mini

# 此处可填写运行程序的命令
# 测试flops sandwich 

# 此处可填写运行程序的命令
python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 8 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-flops-sandwich \
  --log_freq 50 \
  --image_dir /data/public/imagenet-mini \
  # --resume checkpoints/res48-autoslim \


# run1 搞错了，是普通的调用ofa的接口，写的有问题。
# run2 真正的baseline, flops sandwich 
