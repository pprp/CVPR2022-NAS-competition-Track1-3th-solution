#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J supnet      # 作业名字
#SBATCH --gres=gpu:4   # 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load anaconda/2020.11 cuda/10.2 \
 cudnn/7.6.5.32_cuda10.2 nccl/2.9.6-1_cuda10.2
source activate pp

# 此处可填写运行程序的命令
# cd  CVPR_2022_Track1_demo
#cd  AutoDL-Projects

python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 2 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-depth \
  --log_freq 1 \
  --resume checkpoints/res48-depth \
  --image_dir /data/public/imagenet2012 

#python3 test.py
#python3 -m torch.distributed.launch --nproc_per_node 8  ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path /data/public/cifar/ --algo darts-v1
