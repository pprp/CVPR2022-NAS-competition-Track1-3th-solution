import paddle
from model import Model


class FlopsPartition(object):
    FLOPS_LIST = [946450399, 2032925160, 3119399920, 4205874680, 5292349440]
    FIRST_EPOCH = True
    WARMUP_STEP = 1000

    def __init__(self, *args, **kwargs):
        self.partition_info = {
            1: {"teacher_arch": "1346455555550000777777770077777777777700007777777700"},
            2: {"teacher_arch": "1558533333333333755555555577777777777777777777777777"},
            3: {"teacher_arch": "1558533333333333222222222255555555555555552222222222"},
            4: {"teacher_arch": "1558511111111111111111111111111111111111111111111111"},  # 最大网络
        }

    def get_partition_num(self, flops):
        for i in range(len(self.FLOPS_LIST) - 1):
            if flops > self.FLOPS_LIST[i] and flops <= self.FLOPS_LIST[i + 1]:
                return i + 1

    def get_arch_flops(self, arch):
        net = Model(arch=arch, block='basic')
        flops = paddle.flops(net, [1, 3, 224, 224])
        del net
        return flops

    def get_arch_partition_num(self, arch):
        flops = self.get_arch_flops(arch)
        # print("arch:", arch, "flops:", flops)
        num = self.get_partition_num(flops)
        return num
