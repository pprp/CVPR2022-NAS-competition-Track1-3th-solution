import os

import paddle
import paddle.nn as nn
from paddle.nn import CrossEntropyLoss
from paddle.vision.transforms import (
    RandomHorizontalFlip, RandomResizedCrop, SaturationTransform,
    Compose, Resize, HueTransform, BrightnessTransform, ContrastTransform,
    RandomCrop, Normalize, RandomRotation, CenterCrop)
from paddle.io import DataLoader
from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, LinearWarmup

from hnas.utils.callbacks import LRSchedulerM, MyModelCheckpoint
from hnas.utils.transforms import ToArray
from hnas.dataset.random_size_crop import MyRandomResizedCrop
from paddle.vision.datasets import DatasetFolder

from paddleslim.nas.ofa.convert_super import Convert, supernet
from paddleslim.nas.ofa import RunConfig, DistillConfig, ResOFA
from paddleslim.nas.ofa.utils import utils

import paddle.distributed as dist
from hnas.utils.yacs import CfgNode
from hnas.models.builder import build_classifier

def _loss_forward(self, input, tea_input=None, label=None):
    if tea_input is not None and label is not None:
        # cross entropy + knowledge distillation
        ce = paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=self.soft_label,
            axis=self.axis,
            name=self.name)

        kd = paddle.nn.functional.cross_entropy(
            input,
            paddle.nn.functional.softmax(tea_input),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=True,
            axis=self.axis)
        return ce, kd
    elif tea_input is not None and label is None:
        # inplace distillation
        kd = paddle.nn.functional.cross_entropy(
            input,
            paddle.nn.functional.softmax(tea_input),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=True,
            axis=self.axis)
        return kd
    elif label is not None:
        # normal cross entropy
        # print("line 62: ", input.shape, label.shape)
        ce = paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=False,
            axis=self.axis,
            name=self.name)
        return ce
    else:
        raise "Not Implemented Loss."

CrossEntropyLoss.forward = _loss_forward

def _compute(self, pred, tea_pred, label=None, *args):
    if label is None:
        label = tea_pred
    pred = paddle.argsort(pred, descending=True)
    pred = paddle.slice(
        pred, axes=[len(pred.shape) - 1], starts=[0], ends=[self.maxk])
    if (len(label.shape) == 1) or \
        (len(label.shape) == 2 and label.shape[-1] == 1):
        # In static mode, the real label data shape may be different
        # from shape defined by paddle.static.InputSpec in model
        # building, reshape to the right shape.
        label = paddle.reshape(label, (-1, 1))
    elif label.shape[-1] != 1:
        # one-hot label
        label = paddle.argmax(label, axis=-1, keepdim=True)
    correct = pred == label
    return paddle.cast(correct, dtype='float32')

paddle.metric.Accuracy.compute = _compute

from hnas.utils.hapi_wrapper import Trainer

def run(
    backbone='resnet48',
    image_size='224',
    max_epoch=120,
    lr=0.0025,
    weight_decay=0.,
    momentum=0.9,
    batch_size=80,
    dyna_batch_size=4,
    warmup=2,
    phase=None,
    task='expand_ratio',
    resume=None,
    pretrained='checkpoints/resnet48.pdparams',
    image_dir='/root/paddlejob/workspace/env_run/data/ILSVRC2012/',
    save_dir='checkpoints/res48-depth',
    save_freq=20,
    log_freq=100,
    **kwargs
    ):
    run_config = locals()
    run_config.update(run_config["kwargs"])
    del run_config["kwargs"]
    config = CfgNode(run_config)
    config.image_size_list = [int(x) for x in config.image_size.split(',')]

    nprocs = len(paddle.get_cuda_rng_state())
    gpu_str = []
    for x in range(nprocs):
        gpu_str.append(str(x))
    gpu_str = ','.join(gpu_str)
    print(f'gpu num: {nprocs}')
    dist.spawn(main, args=(config,), nprocs=nprocs, gpus=gpu_str)


def main(cfg):
    paddle.set_device('gpu:{}'.format(dist.ParallelEnv().device_id))
    if dist.get_rank() == 0:
        print(cfg)
    IMAGE_MEAN = (0.485,0.456,0.406)
    IMAGE_STD = (0.229,0.224,0.225)

    cfg.lr = cfg.lr * cfg.batch_size * dist.get_world_size() / 256
    warmup_step = int(1281024 / (cfg.batch_size * dist.get_world_size())) * cfg.warmup

    # data augmentation
    transforms = Compose([
        MyRandomResizedCrop(cfg.image_size_list),
        RandomHorizontalFlip(),
        ToArray(),
        Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    val_transforms = Compose([Resize(256), CenterCrop(224), ToArray(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    train_set = DatasetFolder(os.path.join(cfg.image_dir, 'train'), transform=transforms)
    val_set = DatasetFolder(os.path.join(cfg.image_dir, 'val'), transform=val_transforms)
    callbacks = [LRSchedulerM(),
                 MyModelCheckpoint(cfg.save_freq, cfg.save_dir, cfg.resume, cfg.phase),
                 paddle.callbacks.VisualDL(log_dir='./visualdl_log/flops_sandwich'),]

    # build resnet48 and teacher net
    net = build_classifier(cfg.backbone, pretrained=cfg.pretrained, reorder=True)
    tnet = build_classifier(cfg.backbone, pretrained=cfg.pretrained, reorder=False)
    origin_weights = {}
    for name, param in net.named_parameters():
        origin_weights[name] = param

    # convert resnet48 to supernet
    sp_model = Convert(supernet(expand_ratio=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7])).convert(net)  # net转换成supernet
    utils.set_state_dict(sp_model, origin_weights)  # 重新对supernet加载数据
    del origin_weights

    # set candidate config
    cand_cfg = {
            'i': [224],  # image size
            'd': [(2, 5), (2, 5), (2, 8), (2, 5)],  # depth
            'k': [3],  # kernel size
            'c': [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7] # channel ratio
    }

    # default_run_config = {
    #     'train_batch_size': cfg.batch_size,
    #     'n_epochs': [[1], [2, 3], [4, 5]],
    #     'total_images': 12,
    #     'elastic_depth': (2, 3, 4, 5),
    #     'dynamic_batch_size': [1, 1, 1],
    # }

    default_distill_config = {
        'lambda_distill': 0.5,
        'teacher_model': tnet,
        'mapping_layers': None,
        'teacher_model_path': None,
        'distill_fn': None,
        'mapping_op': 'conv2d'
    }

    ofa_net = ResOFA(sp_model,
                    #  run_config=RunConfig(**default_run_config),
                     distill_config=DistillConfig(**default_distill_config), # lambda_distill=1.0
                     candidate_config=cand_cfg,
                     block_conv_num=2)

    ofa_net.set_task(cfg.task, cfg.phase)

    run_config = {'dynamic_batch_size': cfg.dyna_batch_size}
    model = Trainer(ofa_net, cfg=run_config)

    # calculate loss by ce
    model.prepare(
        paddle.optimizer.Momentum(
            learning_rate=LinearWarmup(
                CosineAnnealingDecay(cfg.lr, cfg.max_epoch), warmup_step, 0., cfg.lr),
            momentum=cfg.momentum,
            parameters=model.parameters(),
            weight_decay=cfg.weight_decay),
        CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1,5)))

    model.fit(
        train_set,
        None,
        epochs=cfg.max_epoch,
        batch_size=cfg.batch_size,
        save_dir=cfg.save_dir,
        save_freq=cfg.save_freq,
        log_freq=cfg.log_freq,
        shuffle=True,
        num_workers=1,
        verbose=2, 
        drop_last=True,
        callbacks=callbacks,
    )

    model.evaluate(val_set, batch_size=cfg.batch_size, num_workers=1, eval_sample_num=3)


if __name__ == '__main__':
    import fire
    fire.Fire({"run": run})
