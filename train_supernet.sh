#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J flops9      # 作业名字

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11 
module load cuda/11.1 
module load cudnn/8.1.0.77_CUDA11.1
module load nccl/2.9.6-1_cuda11.0
source activate pp22

# 将数据加载到内存中
# mkdir -p /dev/shm/imagenet2012
# tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012
# tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012

# image_dir=/dev/shm/imagenet2012

mkdir -p /dev/shm/imagenet-mini
cp -r /data/public/imagenet-mini/ /dev/shm

image_dir=/dev/shm/imagenet-mini
#image_dir=/data/public/imagenet-mini

# 此处可填写运行程序的命令
# 测试flops sandwich 

# 此处可填写运行程序的命令
python3 -u train_supernet.py run \
  --backbone resnet48_prelu \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 8 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-flops-run9 \
  --visualdl_log visualdl_log/flops_run9 \
  --log_freq 50 \
  --image_dir $image_dir 
  # --resume checkpoints/res48-autoslim \


# run1 搞错了，是普通的调用ofa的接口，写的有问题。
# run2 真正的baseline, flops sandwich 
# run3 修改line 289 训练teacher使用官方提供的预训练模型和GT一同训练。
# run4 开始跑全量35 epoch, dyna bs=4, parition warmup=5, 修改loss, 其实已经不错了。
# run5 开始跑全量35 教师网络不再使用预训练监督，而是仅依靠GT监督，warmup step修改为1000， 跟run4差不多
# run6 跑mini 35e，不更新patition对象教师网络
# run7 跑mini 100e, 不更新patition对象教师网络
# run8 跑全量 70e 

# flops_run1: 重新运行imagenet-mini, 4个分区【存在问题】：第四个分区永远采样不到
# flops_run2: 3个分区，测试是否能采样到第三个分区
# flops_run3: 手动设置三个，进行测试
# flops_run4: 林臻修改之后进行正式运行
# flops_run5: dpj 新方案, 目前来看收敛效果比较差
# flops_run6: dpj dyna-2 的size设置, patition step=500, warmup_step < 100就开始更新教师网络, 4 gpus, 怀疑崩溃原因是最大网络训练次数过多，尽量满足三明治原则。
# flops_run7: dpj dyna=4 设置，partitionstep=500,warmup_step<1 OOM dyna=2
# flops_run8: dpj dyna=4 设置，partitionstep=500,warmup_step<1 OOM dyna=2, learning rate=0.01 gpu=8, OOM , dyna=2
# flops_run9: v2 dyna=8  learning rate=0.001