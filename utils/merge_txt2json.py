import glob 
import json 

result_path = "checkpoints/results/32th_prelu_flops_lam1_sandwich_balance_2times/32th_prelu_flops_lam1_sandwich_balance_2times.json"
result_txts = glob.glob("checkpoints/results/32th_prelu_flops_lam1_sandwich_balance_2times/*.txt")

# analyse txt files line by line 
result_dict = dict() 

for tmp_file in result_txts:
    with open(tmp_file, "r") as f:
        lines = f.readlines() 
        for line in lines:
            archname, strs, top1, top5 = line.split()
            result_dict[archname] = {
                "acc": float(top1),
                "arch": strs,
            }

print(len(result_dict.keys()))

# dump result_dict to result_path 
with open(result_path, "w") as f:
    json.dump(result_dict, f)


