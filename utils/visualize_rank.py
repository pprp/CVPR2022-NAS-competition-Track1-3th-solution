import json
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axisartist.axislines import Axes, AxesZero

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
    idxs = random.sample(list(range(len(lx))), int(len(lx) * 0.3))
    
    new_x, new_y = [], []
    
    for idx in idxs: 
        new_x.append(lx[idx])
        new_y.append(ly[idx])
    
    new_color = [abs(x-y) for x, y in zip(new_x ,new_y)]
    
    plt.figure(dpi=300)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(xtitle, fontsize=21)
    plt.ylabel(ytitle, fontsize=21)
    plt.title("Pearson Coeff. {:.4f}".format(pearson(lx, ly)), fontsize=22)
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

def count_search_space():
    depth_range1 = list(range(2,6))
    depth_range2 = list(range(2,9))
    cnt = 0
    for stage1 in depth_range1:
        for stage2 in depth_range1:
            for stage3 in depth_range2:
                for stage4 in depth_range1:
                    cnt += 7 ** (stage1 + stage2 + stage3 + stage4)
    return cnt 

def sub_plot_lambda():
    plt.figure(figsize=(28, 5.5), dpi=300)
    
    epoch_range = np.array(list(range(70)))
    plt.subplot(1, 4, 1)
    lambda_range = [1 for _ in range(len(epoch_range))]
    plt.plot(epoch_range, lambda_range, linewidth=5.0)
    # ax = fig.add_subplot(axes_class=Axes)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set(linestyle="-", linewidth=4)
    ax.spines["bottom"].set(linestyle="-", linewidth=4)

    plt.xticks([0, 50], fontsize=45)
    plt.yticks([0., 1.0, 2.0], fontsize=45)
    plt.title("Constant", fontsize=45)
    
    plt.subplot(1, 4, 2)
    lambda_range = [min(epoch / 10, 2.0) for epoch in epoch_range]
    plt.plot(epoch_range, lambda_range, c='purple', linewidth=5.0)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set(linestyle="-", linewidth=4)
    ax.spines["bottom"].set(linestyle="-", linewidth=4)

    plt.xticks([0, 50], fontsize=45)
    plt.yticks([0., 1.0, 2.0], fontsize=45)
    plt.title("Warmup", fontsize=45)
    
    
    plt.subplot(1, 4, 3)
    lambda_range = [2 * np.sin(np.pi * 0.8 * epoch / 70) for epoch in epoch_range]
    plt.plot(epoch_range, lambda_range, c='green', linewidth=5.0)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set(linestyle="-", linewidth=4)
    ax.spines["bottom"].set(linestyle="-", linewidth=4)

    plt.xticks([0, 50], fontsize=45)
    plt.yticks([0., 1.0, 2.0], fontsize=45)
    plt.title("Cosine", fontsize=45)
 
    
    plt.subplot(1, 4, 4)

    def lambda_scheduler(epoch):
        # four stage 
        if epoch < 5:
            return 0.
        elif epoch >= 5 and epoch < 20:
            return 2. / 15 * epoch - (2. / 3.)
        elif epoch >= 20 and epoch < 50:
            return 2. 
        elif epoch >= 50 and epoch < 70:
            return -0.1 * epoch + 7
        else:
            return 0.
    lambda_range = [lambda_scheduler(epoch) for epoch in epoch_range]
    plt.plot(epoch_range, lambda_range, c='red', linewidth=5.0)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set(linestyle="-", linewidth=4)
    ax.spines["bottom"].set(linestyle="-", linewidth=4)

    plt.xticks([0, 50], fontsize=45)
    plt.yticks([0., 1.0, 2.0], fontsize=45)
    plt.title("MultiStage", fontsize=45)
    
    # plt.figtext(0.47, -0.07, 'Epoch', fontsize=45)
    plt.figtext(0.078, 0.5, r'$\lambda$', va='center', rotation='vertical',fontsize=45)
    plt.savefig("labmda.png")
    

if __name__ == "__main__":
    # sub_plot_lambda()
    path1 = r"checkpoints/prior_zenscore.json"
    path2 = r"checkpoints\83.0_mish_rankloss_zenscore_run4.json"
    xtitle = r"ZenScore Prior Ranking"
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
    
    