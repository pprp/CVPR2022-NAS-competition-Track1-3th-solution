#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J flpsd      # 作业名字

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11 
module load cuda/11.1 
module load cudnn/8.1.0.77_CUDA11.1
module load nccl/2.9.6-1_cuda11.0
source activate pp22

# 将数据加载到内存中
mkdir -p /dev/shm/imagenet2012
tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012
tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012

image_dir=/dev/shm/imagenet2012
# image_dir=/data/public/imagenet-mini

# 此处可填写运行程序的命令
# 测试flops sandwich 

# 此处可填写运行程序的命令
python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 4 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-flops-sandwich-run8 \
  --log_freq 50 \
  --image_dir $image_dir 
  
  #/data/public/imagenet2012 \
  # --resume checkpoints/res48-autoslim \


# run1 搞错了，是普通的调用ofa的接口，写的有问题。
# run2 真正的baseline, flops sandwich 
# run3 修改line 289 训练teacher使用官方提供的预训练模型和GT一同训练。
# run4 开始跑全量35 epoch, dyna bs=4, parition warmup=5, 修改loss, 其实已经不错了。
# run5 开始跑全量35 教师网络不再使用预训练监督，而是仅依靠GT监督，warmup step修改为1000， 跟run4差不多
# run6 跑mini 35e，不更新patition对象教师网络
# run7 跑mini 100e, 不更新patition对象教师网络
# run8 跑全量 70e 
