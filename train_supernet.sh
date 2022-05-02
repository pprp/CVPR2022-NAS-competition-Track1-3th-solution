#!/bin/bash
#SBATCH -N 1     # 需要使用的节点数
#SBATCH -J base      # 作业名字
# 需要使用的卡数

#此处可填写加载程序运行所需环境（根据软件需求，可使用 module load export 等方式加载）
module load cuda/11.0
module load anaconda/2020.11 
module load nccl/2.9.6-1_cuda11.0
module load cudnn/8.1.1.33_CUDA11.0
source activate pp

# python -m pip install paddlepaddle-gpu==2.0.2.post110 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

# 将数据加载到内存中
mkdir -p /dev/shm/imagenet2012
tar -kxf /data/public/imagenet2012/train.tar -C /dev/shm/imagenet2012 & tar -kxf /data/public/imagenet2012/val.tar -C /dev/shm/imagenet2012 

# 此处可填写运行程序的命令

# ignore warning 
python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch 70 \
  --batch_size 256 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 4 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-alphanet \
  --log_freq 50 \
  --visualdl_dir "./visualdl_log/alphanet" \
  --image_dir /dev/shm/imagenet2012 \
  # --resume checkpoints/res48-autoslim4 \

# autoslim3: wd=0 cosine_fix=0.05 max_epoch=100 
# autoslim_baseline: autoslim方法。
# autoslim_alphanet: 使用alphanet蒸馏损失函数。中途出现nan的问题
# autoslim_alphanet2: 针对nan的问题，删除min_lr，将iw_clip由5修改为3
# alphanet: 按照2的设置开始全量数据运行 [jobid]:134598



# 检查：
# 1. 数据集是否是内存
# 2. visualdl 的位置
# 3. 运行的算法是否正确
# 4. 检查max epoch是否设置正确
# 5. 是否设置resume对象？
# 6. 检查num of workers