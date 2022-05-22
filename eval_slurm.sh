#!/bin/bash 

sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_0.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_1.json
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_2.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_3.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_4.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_5.json
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_6.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_7.json 
