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
