import paddle
import paddle.nn as nn


def adjust_bn_according_to_idx(bn, idx):
    bn_weights = bn.parameters()[0]
    bn_bias = bn.parameters()[1]
    bn_weights.set_value(paddle.index_select(bn_weights, idx, 0))
    bn_bias.set_value(paddle.index_select(bn_bias, idx, 0))

    if type(bn) in [nn.BatchNorm1D, nn.BatchNorm2D]:
        bn_mean = bn.parameters()[2]
        bn_var = bn.parameters()[3]
        bn_mean.set_value(paddle.index_select(bn_mean, idx, 0))
        bn_var.set_value(paddle.index_select(bn_var, idx, 0))


def make_divisible(v, divisor, min_val=None):
    """This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


