#!/bin/bash 
# Use two gpus and every gpu process two json

# sbatch eval_supernet.sh checkpoints/jsons/SubTest_8_0.json 
sbatch eval_supernet.sh checkpoints/jsons/SubTest_8_1.json
sbatch eval_supernet.sh checkpoints/jsons/SubTest_8_2.json 
sbatch eval_supernet.sh checkpoints/jsons/SubTest_8_3.json 
sbatch eval_supernet.sh checkpoints/jsons/SubTest_8_4.json 
sbatch eval_supernet.sh checkpoints/jsons/SubTest_8_5.json
sbatch eval_supernet.sh checkpoints/jsons/SubTest_8_6.json 
sbatch eval_supernet.sh checkpoints/jsons/SubTest_8_7.json 


# sbatch eval_supernet.sh checkpoints/standalone/CVPR_PDL_test_1.json 