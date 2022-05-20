import json 

NUM_SPLITS = 16

origin_json = "./checkpoints/CVPR_2022_NAS_Track1_test.json"

splited_jsons = [
    f"./checkpoints/jsons_16/SubTest_{NUM_SPLITS}_{i}.json" for i in range(NUM_SPLITS)
]

fin = open(origin_json, "r")
origin_dict = json.load(fin)
splited_dict = [dict() for i in range(NUM_SPLITS)]

fout = [open(splited_jsons[i], "w") for i in range(NUM_SPLITS)]

for idx, (key, value) in enumerate(origin_dict.items()):
    true_idx = idx // (45000 // NUM_SPLITS)
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