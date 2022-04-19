import glob 
import json 

test_path = "./checkpoints/CVPR_2022_NAS_Track1_test.json"
result_path = "./checkpoints/results/results.json"
result_txts = glob.glob("./checkpoints/results/*.txt")

# mapping from arch config to arch name 
str2name = dict() 
with open(test_path, "r") as f:
    test_dict = json.load(f)
    for k, v in test_dict.items():
        str2name[v["arch"]] = k 

# analyse txt files line by line 
result_dict = dict() 
for tmp_file in result_txts:
    with open(tmp_file, "r") as f:
        lines = f.readlines() 
        for line in lines:
            strs, top1, top5 = line.split()
            if strs in str2name.keys():
                archname = str2name[strs]
            else:
                print(f"{strs} do not exits.")
                continue
            result_dict[archname] = {
                "acc": float(top1),
                "arch": strs,
            }

# dump result_dict to result_path 
with open(result_path, "w") as f:
    json.dump(result_dict, f)


