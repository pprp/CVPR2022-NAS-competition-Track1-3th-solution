
import json 
import os 
import random 
import matplotlib.pyplot as plt 
import numpy as np 

def hamming_distance(arch1:str, arch2:str):
    assert len(arch1) == len(arch2)
    # 1 2584 33333000000222222321215451545553545157262723200
    
    arch1 = arch1[5:]
    arch2 = arch2[5:]
    
    dis=0

    for i in range(len(arch1)):
        p1 = int(arch1[i])
        p2 = int(arch2[i])
        dis += 1 if p1 != p2 else 0 
    
    return dis 

def sample_times(all_dict: dict, times:int):
    dis_list = []

    for _ in range(times):
        sampled_keys = random.sample(all_dict.keys(), 2)
        sampled_arch1 = all_dict[sampled_keys[0]]["arch"] 
        sampled_arch2 = all_dict[sampled_keys[1]]["arch"] 
        dis_list.append(hamming_distance(sampled_arch1, sampled_arch2))

    return dis_list

def test_sample_time(all_dict: dict):
    # 33333000000222222321215451545553545157262723200
    anchor_arch = all_dict[random.sample(all_dict.keys(), 1)[0]]["arch"]
    
    cnt = 0 
    dis = 0
    while dis < 30: 
        dis = hamming_distance(anchor_arch, all_dict[random.sample(all_dict.keys(), 1)[0]]["arch"])
        cnt += 1
    print(cnt)

if __name__ == "__main__":
    json_path = "checkpoints/CVPR_2022_NAS_Track1_test.json"
    with open(json_path, "r") as f:
        all_dict = json.load(f)
    
    dis_list = sample_times(all_dict, 1000)

    buck = {}

    for dis in dis_list: 
        if dis not in buck.keys():
            buck[dis] = 1 
        else:
            buck[dis] += 1
    x = []
    y = []
    for k,v in buck.items():
        x.append(k)
        y.append(v)

    # print(f"mean: {np.mean(np.array(dis_list))}")
    # print(f"var:  {np.var(np.array(dis_list))}")
    
    plt.bar(x, y)
    plt.savefig("./hamming_dis.png")

    test_sample_time(all_dict)

    
