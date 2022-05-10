import json 
import os 
import glob 

whole_json_path = "checkpoints/CVPR_2022_NAS_Track1_test.json"
txts_path = "checkpoints/results/res48-flops-run5"
full_txts = glob.glob(os.path.join(txts_path, "*.txt")) 
save_json_path = "checkpoints/results/CVPR_Track1_left.json"

arch2config = {}
with open(whole_json_path, "r") as f:
    archdict = json.load(f)
    for k, v in archdict.items():
        arch2config[k] = v["arch"]



processed_archname = []

for single_txt in full_txts:
    # full_txt_path = os.path.join(txts_path, single_txt)
    with open(single_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            archname, config, acc1, acc2 = line.split()
            processed_archname.append(archname)

print(len(arch2config.keys()))
print(len(processed_archname))


untested_archs = set(arch2config.keys()) - set(processed_archname)

new_dict = {}

for arch in untested_archs:
    new_dict[arch] = {
        "acc": 0.,
        "arch": arch2config[arch],
    }

with open(save_json_path, "w") as f:
    json.dump(new_dict, f)