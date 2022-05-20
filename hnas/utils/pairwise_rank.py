import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class PairwiseRankLoss(nn.Layer):
    """pairwise ranking loss for rank consistency 
        flops1: the flops of arch1 
        flops2: the flops of arch2 
        loss1: the batch loss of arch1 
        loss2: the batch loss of arch2 
    """
    def forward(self, flops1, flops2, loss1, loss2, coeff=1.):
        return coeff * F.relu(loss2-loss1.detach()) if flops1 < flops2 else coeff * F.relu(loss1.detach()-loss2)