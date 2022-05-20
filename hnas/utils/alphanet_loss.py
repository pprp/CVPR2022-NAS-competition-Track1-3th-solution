import paddle
from paddle import nn


class CrossEntropyLossSoft(nn.Layer):
    """ inplace distillation for image classification """

    def forward(self, output, target):
        output_log_prob = nn.functional.log_softmax(output, axis=1)
        target = nn.functional.one_hot(paddle.cast(target,dtype='int64'), num_classes=output.shape[1])
        target = paddle.cast(target, dtype='float32')
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -paddle.bmm(target, output_log_prob)
        return cross_entropy_loss.mean()


class KLLossSoft(nn.Layer):
    """ inplace distillation for image classification
            output: output logits of the student network
            target: output logits of the teacher network
            T: temperature
            KL(p||q) = Ep \log p - \Ep log q
    """

    def forward(self, output, soft_logits, target=None, temperature=1., alpha=0.9):
        output, soft_logits = output / temperature, soft_logits / temperature
        soft_target_prob = nn.functional.softmax(soft_logits, axis=1)
        output_log_prob = nn.functional.log_softmax(output, axis=1)
        kd_loss = -paddle.sum(soft_target_prob * output_log_prob, axis=1)
        if target is not None:
            n_class = output.size(1)
            target = nn.functional.one_hot(paddle.cast(target,dtype='int64'), num_classes=n_class)
            target = paddle.cast(target, dtype='float32')
            output_log_prob = output_log_prob.unsqueeze(2)
            ce_loss = -paddle.bmm(target, output_log_prob).squeeze()
            loss = alpha * temperature * temperature * kd_loss + (1.0 - alpha) * ce_loss
        else:
            loss = kd_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CrossEntropyLossSmooth(nn.Layer):
    def __init__(self, label_smoothing=0.1, reduction='mean'):
        super(CrossEntropyLossSmooth, self).__init__()
        self.eps = label_smoothing
        self.reduction = reduction

    """ label smooth """
    def forward(self, output, target):
        n_class = output.shape[1]
        one_hot = nn.functional.one_hot(paddle.cast(target, dtype='int64'), num_classes=n_class)
        target = one_hot * (1 - self.eps) + self.eps / n_class
        target = paddle.cast(target, dtype='float32')
        output_log_prob = nn.functional.log_softmax(output, axis=1)
        output_log_prob = output_log_prob.unsqueeze(2)

        loss = -paddle.bmm(target, output_log_prob)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss



def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3):
    assert isinstance(alpha, float)
    q_prob = nn.functional.softmax(q_logits, axis=1).detach()
    p_prob = nn.functional.softmax(p_logits, axis=1).detach()
    q_log_prob = nn.functional.log_softmax(q_logits, axis=1) # gradient is only backpropagated here

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = paddle.clip(importance_ratio, 0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = paddle.pow(importance_ratio, alpha)
        iw_alpha = paddle.clip(iw_alpha, 0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = paddle.sum(q_prob * (f - f_base), axis=1)
    grad_loss = -paddle.sum(q_prob * rho_f * q_log_prob, axis=1)
    # loss = torch.sum(q_prob * (f - f_base), dim=1)
    # grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss


class AdaptiveLossSoft(nn.Layer):
    def __init__(self, alpha_min=-1.0, alpha_max=1.0, iw_clip=3.0, reduction='mean'):
        super(AdaptiveLossSoft, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip
        self.reduction = reduction

    def forward(self, output, target, alpha_min=None, alpha_max=None):
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max

        loss_left, grad_loss_left = f_divergence(output, target, alpha_min, iw_clip=self.iw_clip)
        loss_right, grad_loss_right = f_divergence(output, target, alpha_max, iw_clip=self.iw_clip)

        ind = paddle.greater_than(loss_left, loss_right)
        ind = paddle.cast(ind, dtype='float32')
        loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
