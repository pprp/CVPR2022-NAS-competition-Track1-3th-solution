English | [简体中文](README_ch.md)

# Demo for CVPR 2022 Track1

- [Introduction](#Introduction)
- [TaskDefinition](#TaskDefinition)
- [StandaloneTrainning](#StandaloneTrainning)
- [SupernetTrainning](#SupernetTrainning)

This is standalone and supernet trainning demo for CVPR 2022 Track1. 

# Introduction

Neural Architecture Search (NAS) has been successfully used to automate the design of deep neural network architectures, achieving results that outperform hand-designed models in many computer vision tasks. Parameter sharing based OneshotNAS approaches can significantly reduce the training cost. However, there are still three issues to be urgently solved in the development of lightweight NAS. 

First, the performance of the network sampled from the supernet is inconsistent with the performance of the same network trained independently. This results in an incorrect evaluation and improper ranking of candidate performance. Second, the existing performance prediction benchmarks usually focus on the evaluation of networks from the same search space, while the generalization to difference search spaces has been not explored. This makes the trained prediction model less practical in real-world applications.
Third, the missing of latency evaluation benchmarks hinders the development of lightweight NAS in embedded systems.  

Among which, the consistence issue is one of major promblem of weight-sharing NAS. The performance of the network sampled from the supernet is inconsistent with the performance of the same network trained independently. This results in an incorrect evaluation and improper ranking of candidate performance. Track 1 tries to narrow the performance gap between candidates with the parameters  extracted from the shared parameters and the same architectures with the parameter trained independently. This track requires participants to submit pre-trained supernet using their own strategies. Then, we will evaluate the performance gap between candidates with the parameters  extracted  from the submitted supernet and performances provided by NAS-Bench.  Evaluation metric for track 1. In this track, we utilize Kendall metric, which is a common measurement of the correlation between two ranking, to evaluate the performance gap.  

In this competition, we aim to benchmark lightweight NAS in a systematic and realistic approach. We introduce to make a step forward in advancing the state-of-the-art in the field of lightweight NAS by organising the first challenge and providing the comprehensive benchmarks for fair comparisons. We set up three competition tracks and encourage participants to propose novel solutions to advance the state-of-the-art.    

The challenge will represent the first thorough quantitative evaluation on the topic of lightweight NAS. Furthermore, the competition will explore how far we are from attaining satisfactory NAS results in various scenarios and help the community to understand the performance gap between the state-of-the-art from academic and industry.  **The winner teams are invited to present their solutions on [CVPR 2022 NAS workshop](https://www.cvpr-nas.com/).   Besides, Top 3 teams of each track are invited to submit papers  without using the cmt system（only for extended abstract paper）. Please refer to the [paper submisison page](https://www.cvpr-nas.com/Paper_Submission) for more details .**  

# TaskDefinition

Parameter sharing based OneshotNAS approaches can significantly reduce the training cost. However, there are still three issues to be urgently solved in the development of lightweight NAS. Among which, the consistence issue is one of major promblem of weight-sharing NAS. The performance of the network sampled from the supernet is inconsistent with the performance of the same network trained independently. This results in an incorrect evaluation and improper ranking of candidate performance. Track 1 tries to narrow the performance gap between candidates with the parameters  extracted from the shared parameters and the same architectures with the parameter trained independently. This track requires participants to submit pre-trained supernet using their own strategies. Then, we will evaluate the performance gap between candidates with the parameters  extracted  from the submitted supernet and performances provided by NAS-Bench.  Evaluation metric for track 1. In this track, we utilize Kendall metric, which is a common measurement of the correlation between two ranking, to evaluate the performance gap.  

In this track, the super network builds a search space based on ResNet48. The number of layers of the network and the number of channels in each layer of the network can be searched. The search space is as follows:
Number of network layers:
    1. For the ResNet structure, there are 4 stages in total, and each stage has a different number of blocks;
    2. The 1st, 2nd, and 4th stages have 5 blocks respectively, so the block search space of these three stages is: [2,3,4,5];
    3. The third stage has 8 blocks, and the block search space of this stage is: [2,3,4,5,6,7,8].
The number of channels per layer of the network:
    1. For the ResNet structure, the basic channel numbers corresponding to the four stages are: [64, 128, 256, 512];
    2. The scaling ratio of each conv layer channel is: [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7];
    3. Therefore, the channel search space in each stage is: the number of basic channels x [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7], and the number of channels is an integer multiple of 8;
    4. There is a stem conv between the first conv and the first stage in the ResNet structure, the basic channel number is 64, and the corresponding scaling ratio is still: [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7] , and the number of channels is an integral multiple of 8;
Submodel coding:
We uniformly use the model code to represent the sub-model of the supernet. The model code has a total of 51 bits, for example: 348311131310000332323230014143424241434543757770000, among which:
    1. The sub-model coding length is kept at 51 bits;
    2. The first to fourth digits represent the number of blocks selected in the first four stages;
    3. [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7] The scaling ratios are coded as: 1, 2, 3, 4, 5, 6, 7 respectively. For example, if the encoding bit is 3, then the channel of the conv layer is: the number of basic channels x 0.9, and is a multiple of 8;
    4. The fifth bit represents the encoding corresponding to the scaling ratio of the channel layer of the stem conv layer in ResNet;
    5. For each stage, if the number of blocks is insufficient, the corresponding code will be filled with 0. Since each ResNet block has 2 convs, two zeros are added for each less block.

We will Release  **45,000 sub-networks** from the ResNet48 search space. Participants  are required to use **Imagenet dataset** to train the supernetwork containing all sub-networks of ResNet48 search space. And then evaluate the performance of the 45,000 sub-networks based on **the parameters of the trained super network**. Then, submits the 45,000 sub-networks along with their performances to the server when the learder board is open. The organizer will evaluate the rank consistence based on pre-selected sub-structures (will not release until the competion close) out of 45,000 structures . The rank consistence is based on the Kendall metric.  

The second stage is divided into A/B Leaderboard. The A/B Leaderboard are based on the same submission file submitted by the participants, but the indexes calculating the score are different. Before the submission deadline, only A Leaderboard is visible to the participants. Finally, the B Leaderboard will be announced to the participants. The final ranking of the competition is based on B Leaderboard. To prevent participant cheating, the metric of leaderboard A list is the absolute value of the Pearson correlation coefficient (which has a strong correlation with Kendall tau), and the metric of leaderboard B is Kendall tau.

**Important instructions for the second stage:**  
Participants **Must comply** the following rules:  
1）Participants can adjust the learning rate, batch size, augmentation, optimizer and other hyperparameters of the supernet training according to the accuracy of the stand alone gt trained by themselves, but the  participants cannot use the accuracy of gt to train the supernet, see 2),3 for details(But not limited to 2), 3)). As long as training process of supernet involves gt, it will be judged as a violation.  
2）Based on the deviation of gt accuracy and supernet accuracy to train  the supernet will be judged as a violation.  
3）Based on the deviation deviation of gt ranking and supernet ranking  to train the supernet will be judged as a violation.  
4）Participants who use any additional training data to train the supernet will be judged as a violation.  
5）Participants who try to attack leaderboard A' indexes will be judged as violations.  
6）The accuracy of the submitted resluts must be based on he supernet. The accuracy obtained by direct training will be judged as a violation, and the accuracy generated by predictor and other methods will be judged as a violation.  
7）Participants can use multiple supernet as supervision when training super networks, such as self-supervision, Siamese supernet, etc., but they can only generate the submitted accuracy based on single supernet.  
8）In the final leaderboard B, the top three teams need to submit code to reproduce the submitted results. The code includes 1) The pre-trained model (supernet) and script that generate the final submission results based on the pre-trained supernet, and 2) The code to reproduce the supernet training process.  
9）Rule 5) violation may be disqualified in advance, other violations will be reviewed in the code review stage after the deadline of leaderboard A . If any of the top three team violate the rules, the final leaderboard B  ranking will be postponed. 

# StandaloneTrainning

```bash
pip install -r requirements.txt
python3 -m paddle.distributed.launch --gpu 0,1,2,3 pretrain.py run --arch 1322221222220000122200000024540000000000005525000000 --image_dir /root/paddlejob/workspace/env_run/data/ILSVRC2012/ --batch_size 1024 --max_epoch 90 --warmup 2 > 1322221222220000122200000024540000000000005525000000.log
``` 

# SupernetTrainning
[Pretrained model](https://aistudio.baidu.com/aistudio/datasetdetail/134077)
```bash
sh train_supernet.sh
``` 
