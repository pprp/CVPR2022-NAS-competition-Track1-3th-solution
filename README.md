
# Rank Consistency NAS for CVPR21 workshop

这是CVPR 2022 NAS workshop Track1独立训练子网络代码以及supernet训练代码

## 文件结构

```
- checkpoints 保存权重, json文件
- hnas 提供了独立训练的代码以及工具包
- paddleslim 核心算法once for all的实现 
- scripts 训练脚本和验证脚本
- eval_supernet.py 验证45000个子网的代码 
- train_supernet.py 训练超网的代码 
- model.py 配合standalone训练的模型
- standalone.py 独立训练某个网络的代码 
```



## 运行脚本

### Standalone Trainning

```bash
pip install -r requirements.txt
python3 -m paddle.distributed.launch --gpu 0,1,2,3 standalone.py run --arch 1322221222220000122200000024540000000000005525000000 --image_dir /root/paddlejob/workspace/env_run/data/ILSVRC2012/ --batch_size 1024 --max_epoch 90 --warmup 2 > 1322221222220000122200000024540000000000005525000000.log
``` 

### Supernet Train

[预训练模型地址](https://aistudio.baidu.com/aistudio/datasetdetail/134077)

```bash
sbatch scrips/train_supernet.sh
``` 

### Supernet Eval

```bash
sbatch scrips/eval_supernet.sh
``` 


## 背景

本次比赛分为两个赛道，赛道一为[超网络赛道](https://aistudio.baidu.com/aistudio/competition/detail/149/0/introduction)，旨在解决OneshotNAS的一致性问题；赛道二为[模型性能预测赛道](https://aistudio.baidu.com/aistudio/competition/detail/150/0/introduction)，旨在不做任何训练的情况，准确的预测任意模型结构在特定评测集的性能。**获胜的队伍会被邀请在[CVPR 2022 NAS workshop](https://www.cvpr-nas.com/)上宣讲队伍的技术方案。 此外，各Track 前三名会被邀请提交论文（extended abstract论文可以不通过cmt系统提交，regular论文需要系统提交) ，论文要求详见[CVPR NAS workshop论文提交页面](https://www.cvpr-nas.com/Paper_Submission)**

## 赛题表述

本赛道旨在解决超网络的一致性问题。基于超网络性能与独立训练子网络性能最一致的队伍将获得冠军。  

在本赛道中，超网络基于ResNet48构建搜索空间，网络的层数、网络每层的通道数可以搜索，搜索空间如下：
网络层数：
    1.    对于ResNet结构，共4个stage，每个stage有不同的block数；
    2.    第1、2、4 stage分别有5个block，故这三个stage的block搜索空间为：[2,3,4,5]；
    3.    第3个stage有8个block，此stage的block搜索空间为：[2,3,4,5,6,7,8]。
网络每层通道数：
    1.    对于ResNet结构，4个stage分别对应的基本通道数为：[64， 128，256，512]；
    2.    每个conv层通道放缩比例的有：[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]；
    3.    故每个stage中的通道搜索空间为：基本通道数 x [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]，且通道数是8的整倍数；
    4.    ResNet结构中第一个conv到第一个stage之间有一个stem conv，基本通道数位64，与其对应的放缩比例仍为：[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]，且通道数是8的整倍数；

子模型编码：

我们统一使用模型编码表示超网的子模型，模型的编码共51位，比如：

348311131310000332323230014143424241434543757770000，其中：
    1.    子模型编码长度保持为51位；
    2.    第1～4位数字分别表示前4个stage被选取的block数量；
    3.    [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]放缩比例分别编码为：1、2、3、4、5、6、7。比如如果编码位3，那么该conv层的channel为：基本通道数 x 0.9，且为8的倍数；
    4.    第5位表示ResNet中stem conv层的通道层的放缩比例对应的编码；
    5.    对于每个stage中，block数量不足，对应的编码会补0。由于每个ResNet block有2个conv，所以每少一个block，补2个0。

近期，我们会Release ResNet48搜索空间中选择**45000个子网络**, 参赛选手需要使用**ImageNet数据集**训练包含ResNet48搜索空间的超网络，然后基于训练好**超网络的参数**评估这45000个子网络的性能，并在榜单提交入口开放之后，将45000个模型结构与这些结构对应的性能提交的服务器。主办方会基于45000个结构中的预先选定的若干个子结构（比赛结束前对参赛者不可见）的排序一致性(Kendall metric)来评估参赛选手的成绩。  

比赛分A/B榜单，A/B榜单都基于选手提交的同一份提交文件，但是计算分数的节点的编号不同。比赛提交截止日期前仅A榜对选手可见，比赛结束后B榜会对选手公布，比赛最终排名按照选手成绩在B榜的排名。为防止选手使用目标结构撞库，A榜榜单指标为Pearson相关系数（和Kendall tau有强相关性）的绝对值，B榜榜单指标为Kendall tau

**重要说明：**  
请参赛选手**务必遵守**以下几点规则:  

1）参赛选手可以根据自己训练的stand alone ground truth (gt) 的精度，调节supernet训练的学习率，batch size, augmentation, 优化器等超参数，但是选手不能使用gt的精度训练超网络, 具体见2)，3)但不仅限于2)，3）只要gt参与到supernet的训练过程就会被判定违规  
2) 基于gt精度与supernet精度的偏差对supernet反向传播会被判定违规  
3) 基于gt的ranking与基于supernet ranking偏差对supernet反向传播会被判定违规  
4) 选手使用任何额外的训练数据训练supernet会被判定违规  
5) 选手通过大量的低精度/固定精度或0精度结构结合少量目标结构的方式对A榜单gt编号进行撞库会被判定违规  
6) 选手最终提交的子网络的精度必须是基于超网络的精度，提交直接训练得到的精度违规，提交经过predictor等方式生成的精度违规  
7) 选手在训练超网络的时候可以使用多个超网络作为监督，比如自监督，孪生超网络等方式，但是最终只能基于1个超网络生成子网络的精度  
8) 最终B榜单前三名选手需要提交代码来复现提交结果，代码包含1)基于训练好的supernet生成选手最终提交结果的模型及脚本，2）复现supernet训练过程的脚本  
9) 第5)点违规可能被提前取消参赛资格，除5)以外违规会在A榜截止后的代码review阶段审核，如果前三名存在违规则最终B榜单排名顺延  

