from model import Model
import paddle 

def get_arch_flops(arch):
    net = Model(arch=arch, block='basic')
    flops = paddle.flops(net, [1, 3, 224, 224])
    del net
    return flops