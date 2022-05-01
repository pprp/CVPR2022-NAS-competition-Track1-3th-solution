import json  
import os 
import glob
from unittest import result 

json_path = "./checkpoints/CVPR_2022_NAS_Track1_test.json"
save_path = "./checkpoints/untested/untestcalibn_8_8.json"
txt_path = "checkpoints/results/calibrationbn"

with open(json_path, "r") as f:
    arch_dict = json.load(f)

all_arch_list = []
computed_arch_list = []

archname2arch = {}

for key, value in arch_dict.items():
    all_arch_list.append(key)
    archname2arch[key] = value["arch"]

result_txts = glob.glob("./checkpoints/results/calibrationbn/*.txt")

for tmp_file in result_txts:
    with open(tmp_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            archname, strs, top1, top5 = line.split()
            computed_arch_list.append(archname)

uncomputed_arch_list = set(all_arch_list) - set(computed_arch_list)
print(len(uncomputed_arch_list))

uncomputed_dict = {}

for archname in uncomputed_arch_list:
    uncomputed_dict[archname] = {
        "acc":0, 
        "arch": archname2arch[archname],
    }

with open(save_path, "w") as f:
    json.dump(uncomputed_dict, f)
