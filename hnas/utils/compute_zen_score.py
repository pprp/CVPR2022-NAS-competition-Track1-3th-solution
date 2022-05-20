'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import paddle
import numpy as np
import paddle.nn as nn 

from paddle.nn.initializer import Constant, Normal



def compute_nas_score(model, batch_size, resolution = 224, repeat= 32,  mixup_gamma=1e-2):
    nas_score_list = []
    for repeat_count in range(repeat):

        #network_weight_gaussian_init(model)
        for sub_layer in model.sublayers():
            if isinstance(sub_layer, nn.Conv2D):
                Normal(mean=0., std=1.0)(sub_layer.weight)
                if hasattr(sub_layer, 'bias') and sub_layer.bias is not None:
                    Constant(0.)(sub_layer.bias)
                    
            elif isinstance(sub_layer, nn.BatchNorm2D):               
                Constant(1.)(sub_layer.weight)
                Constant(0.)(sub_layer.bias)

            elif isinstance(sub_layer, nn.Linear):
                Normal(mean=0., std=1.0)(sub_layer.weight)
                if hasattr(sub_layer, 'bias') and sub_layer.bias is not None:
                    Constant(0.)(sub_layer.bias)
            else:
                continue                                
        
        input = paddle.randn(shape=[batch_size, 3, resolution, resolution], dtype='float32')
        input2 = paddle.randn(shape=[batch_size, 3, resolution, resolution], dtype='float32')
        
        mixup_input = input + mixup_gamma * input2
        
        # output = model.forward_pre_GAP(input)
        output = model(input)
        mixup_output = model(mixup_input)
                
        nas_score = paddle.sum(paddle.abs(output - mixup_output), axis=[1, 2, 3])
        nas_score = paddle.mean(nas_score)
        
        # compute BN scaling
        log_bn_scaling_factor = 0.0
        for sub_layer in model.sublayers():
            if isinstance(sub_layer, nn.BatchNorm2D):
                bn_scaling_factor = paddle.sqrt(paddle.mean(sub_layer._variance))
                log_bn_scaling_factor += paddle.log(bn_scaling_factor)
            pass
        pass        
        
        nas_score = paddle.log(nas_score) + log_bn_scaling_factor
        nas_score_list.append(float(nas_score))

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)

    return float(avg_nas_score), float(std_nas_score), float(avg_precision)

'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import global_utils, argparse, ModelLoader, time

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def compute_nas_score(gpu, model, mixup_gamma, resolution, batch_size, repeat, fp16=False):
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            input = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            input2 = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            output = model.forward_pre_GAP(input)
            mixup_output = model.forward_pre_GAP(mixup_input)

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))


    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    return info


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)
    the_model = ModelLoader.get_model(opt, sys.argv)
    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)


    start_timer = time.time()
    info = compute_nas_score(gpu=args.gpu, model=the_model, mixup_gamma=args.mixup_gamma,
                             resolution=args.input_image_size, batch_size=args.batch_size, repeat=args.repeat_times, fp16=False)
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info['avg_nas_score']
    print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')

'''
