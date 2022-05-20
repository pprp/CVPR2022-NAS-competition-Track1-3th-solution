#!/bin/bash 
# Use two gpus and every gpu process two json

# sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_0.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_1.json
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_2.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_3.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_4.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_5.json
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_6.json 
sbatch --gpus=1 eval_supernet.sh checkpoints/jsons/SubTest_8_7.json 


#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_0.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_1.json
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_2.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_3.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_4.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_5.json
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_6.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_7.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_8.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_9.json
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_10.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_11.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_12.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_13.json
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_14.json 
#sbatch eval_supernet.sh checkpoints/jsons_16/SubTest_16_15.json 
