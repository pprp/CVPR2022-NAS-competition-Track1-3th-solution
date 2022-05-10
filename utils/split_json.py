import json 

NUM_SPLITS = 8

origin_json = "checkpoints/results/CVPR_Track1_left.json"

splited_jsons = [
    f"./checkpoints/jsons_left/SubTest_{NUM_SPLITS}_{i+8}.json" for i in range(NUM_SPLITS)
]

fin = open(origin_json, "r")
origin_dict = json.load(fin)
splited_dict = [dict() for i in range(NUM_SPLITS)]

fout = [open(splited_jsons[i], "w") for i in range(NUM_SPLITS)]

for idx, (key, value) in enumerate(origin_dict.items()):
    true_idx = idx // (len(origin_dict.keys()) // NUM_SPLITS)
    print(f"idx : {idx} true_idx: {true_idx}")
    if true_idx == NUM_SPLITS:
        true_idx -= 1
    splited_dict[true_idx][key] = value 

print(len(splited_dict[0].keys()))

for i in range(NUM_SPLITS):
    json.dump(splited_dict[i], fout[i])

fin.close()
for i in range(NUM_SPLITS):
    fout[i].close() 