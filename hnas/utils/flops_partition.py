import paddle
from model import Model


class FlopsPartition(object):
    FLOPS_LIST = [946450399, 2465677200, 5292349440]
    WARMUP_STEP = 500

    def __init__(self, *args, **kwargs):
        # middle teacher network and max teacher network
        self.partition_info = {
            1: {"teacher_arch": "1336355535150000735333000072625242321200007555350000"},
            2: {"teacher_arch": "1558511111111111111111111111111111111111111111111111"},  
        }
    
    def update_teacher_arch(self, arch):
        # won't update max teacher network
        self.partition_info[1]["teacher_arch"] = arch 

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
        num = self.get_partition_num(flops)
        return num

if __name__ == "__main__":
    m = FlopsPartition()
    # arch = "1226333313230000332300000052121222124232424535550000"
    arch = "1558533333333333755555555577777777777711111111111111"
    print(m.get_arch_flops(arch))
    print(m.get_arch_partition_num(arch))
