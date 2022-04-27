import random 
import numpy as np 


# candidate_ratio = [1,2,3,4,5,6,7]
# #[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

# # sample depth random 
# depth_list = []
# depth_range = [(2,5), (2,5), (2,8), (2,5)]
# for i in range(len(depth_range)):
#     depth_list.append(random.randint(depth_range[i][0], depth_range[i][1]))

# print(f"random depth list: {depth_list}")

# # sample ratio according to depth 
# ratio_list = []
# ratio_list.append(random.sample(candidate_ratio, len(candidate_ratio))) # stem conv 
# for i in range(len(depth_range)):
#     # stage i = 0,1,2,3 
#     depth_stage = depth_list[i]
#     sc_channel = random.sample(candidate_ratio, 1)

#     for j in range(depth_range[i][1]): # 5 5 8 5 
#         # sample the second ratio for build_ss (shortcut)
#         if j < depth_stage:
#             ratio_list.append(random.sample(candidate_ratio, len(candidate_ratio)))
#             ratio_list.append(sc_channel * len(candidate_ratio))
#         else:
#             ratio_list.append([0] * len(candidate_ratio))
#             ratio_list.append([0] * len(candidate_ratio))

# def convert_list2str(lst):
#     result_str = ""
#     for num in lst:
#         result_str += str(num)
#     return result_str 

# ratio_list = np.transpose(ratio_list)



# print(configs_list)



candidate_ratio = [1,2,3,4,5,6,7]

def convert_list2str(lst):
    result_str = ""
    for num in lst:
        result_str += str(num)
    return result_str 

# sample depth random 
depth_list = []
depth_range = [(2,5), (2,5), (2,8), (2,5)]
for i in range(len(depth_range)):
    depth_list.append(random.randint(depth_range[i][0],depth_range[i][1]))

# sample ratio according to depth 
ratio_list = []
first_stem = random.sample(candidate_ratio, 1)
ratio_list.append(first_stem * len(candidate_ratio)) # stem conv 
for i in range(len(depth_range)):
    # stage i = 0,1,2,3 
    depth_stage = depth_list[i]
    # sample the second ratio for build_ss (shortcut)
    sc_channel = first_stem if i == 0 else random.sample(candidate_ratio, 1) 
    for j in range(depth_range[i][1]): # 5 5 8 5 
        if j < depth_stage:
            ratio_list.append(random.sample(candidate_ratio, len(candidate_ratio)))
            ratio_list.append(sc_channel * len(candidate_ratio))
        else:
            ratio_list.append([0] * len(candidate_ratio))
            ratio_list.append([0] * len(candidate_ratio))

ratio_list = np.transpose(ratio_list)

configs_list = [convert_list2str([1] + depth_list + list(cfg)) for cfg in ratio_list]

print(configs_list)