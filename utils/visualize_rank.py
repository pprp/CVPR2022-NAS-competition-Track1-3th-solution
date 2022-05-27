import json
import numpy as np
import matplotlib.pyplot as plt
import random 
from kendall import pearson

# plt.style.use('ggplot')

def data_generate(prior_flops_info, prior_zenscore_info):
    lx = []
    ly = []
    for arch in prior_flops_info.keys():
        lx.append(prior_flops_info[arch])
        ly.append(prior_zenscore_info[arch])
        
    return lx, ly

def draw_rank_illustration(lx:list, 
                   ly:list, 
                   xtitle:str="FLOPs Prior Ranking", 
                   ytitle:str="ZenScore Prior Ranking"):
    # heatmap, xedges, yedges = np.histogram2d(lx, ly, bins=(100, 100))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # plt.clf()
    idxs = random.sample(list(range(len(lx))), int(len(lx) * 0.5))
    
    new_x, new_y = [], []
    
    for idx in idxs: 
        new_x.append(lx[idx])
        new_y.append(ly[idx])
    
    new_color = [abs(x-y) for x, y in zip(new_x ,new_y)]
    
    plt.figure(dpi=300)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title("Pearson rank correlation: {:.4f}".format(pearson(lx, ly)))
    plt.scatter(new_x, new_y, c=new_color, s=5, marker='.', cmap=plt.cm.get_cmap('viridis'), alpha=0.9)


    # plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.savefig('compare_filter_res_heatmap.jpg')
    # plt.show()

def convert_sort_info(prior_dict):
    """add sort rank"""
    arch2name = {}    
    for k, v, in prior_dict.items():
        arch2name[v['arch']] = k    
    def get_prior(pd):
        return float(pd['acc'])    
    sorted_list = sorted(prior_dict.values(), key=get_prior, reverse=True)    
    result_dict = {}    
    for i, tmp_dict in enumerate(sorted_list):
        result_dict[arch2name[tmp_dict["arch"]]] = i     
    return result_dict 

if __name__ == "__main__":
    path1 = r"checkpoints/prior_flops.json"
    path2 = r"checkpoints/26th_mish_zenscore_4stages_max15_sandwich_3times.json"
    xtitle = r"FLOPs Prior Ranking"
    ytitle = r"ZenScore Guided Ranking"
        
    with open(path1) as f:
        prior_flops_info = json.load(f)

    with open(path2) as f:
        prior_zenscore_info = json.load(f)
        
    prior_flops_info = convert_sort_info(prior_flops_info)
    prior_zenscore_info = convert_sort_info(prior_zenscore_info)
    
    lx, ly = data_generate(prior_flops_info, prior_zenscore_info)
    
    draw_rank_illustration(lx, ly, 
                           xtitle=xtitle,
                           ytitle=ytitle)
    
    