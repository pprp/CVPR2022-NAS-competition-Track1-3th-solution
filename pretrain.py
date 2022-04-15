import os
import paddle
import paddle.distributed as dist

from paddle.nn import CrossEntropyLoss
from paddle.vision.transforms import RandomHorizontalFlip, RandomResizedCrop, Compose, Normalize, CenterCrop, Resize
from paddle.io import DataLoader
from paddle.vision.datasets import DatasetFolder
from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, LinearWarmup

from hnas.utils.transforms import ToArray
from hnas.utils.yacs import CfgNode


from model import Model


def run(
    arch='',
    mode='all',
    max_epoch=90,
    lr=0.1,
    weight_decay=1e-4,
    momentum=0.9,
    batch_size=96,
    warmup=2,
    resume=None,
    phase=None,
    pretrained=False,
    image_dir='/workspace/data/ILSVRC2012/',
    save_dir='checkpoints/',
    save_freq=50,
    log_freq=10,
    **kwargs
    ):
    assert mode in ['all', 'train', 'eval']
    run_config = locals()
    run_config.update(run_config["kwargs"])
    del run_config["kwargs"]
    config = CfgNode(run_config)

    nprocs = len(paddle.get_cuda_rng_state())
    gpu_str = []
    for x in range(nprocs):
        gpu_str.append(str(x))
    gpu_str = ','.join(gpu_str)
    print(f'gpu num: {nprocs}')
    # dist.spawn(main, args=(config,), nprocs=nprocs, gpus=gpu_str)
    main(config)


def main(cfg):
    paddle.set_device('gpu:{}'.format(dist.ParallelEnv().device_id))
    if dist.get_rank() == 0:
        print(cfg)
    IMAGE_MEAN = (0.485,0.456,0.406)
    IMAGE_STD = (0.229,0.224,0.225)

    cfg.lr = cfg.lr * cfg.batch_size * dist.get_world_size() / 256
    warmup_step = int(1281024 / (cfg.batch_size * dist.get_world_size())) * cfg.warmup

    transforms = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToArray(),
        Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    val_transforms = Compose([Resize(256), CenterCrop(224), ToArray(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    train_set = DatasetFolder(os.path.join(cfg.image_dir, 'train'), transform=transforms)
    val_set = DatasetFolder(os.path.join(cfg.image_dir, 'val'), transform=val_transforms)
    cfg.save_dir = os.path.join(cfg.save_dir, str(cfg.arch))
    config = {"i": [224], "d": [[2, 5], [2, 5], [2, 8], [2, 5]], "k": [3], "c": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]}

    net = Model(arch=str(cfg.arch), block='basic')

    model = paddle.Model(net)

    model.prepare(
        paddle.optimizer.Momentum(
            learning_rate=LinearWarmup(
                CosineAnnealingDecay(cfg.lr, cfg.max_epoch), warmup_step, 0., cfg.lr),
            momentum=cfg.momentum,
            parameters=model.parameters(),
            weight_decay=cfg.weight_decay),
        CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5)))
    if cfg.mode in ['all', 'train']:
        model.fit(
            train_set,
            None,
            epochs=cfg.max_epoch,
            batch_size=cfg.batch_size,
            save_dir=cfg.save_dir,
            save_freq=cfg.save_freq,
            log_freq=cfg.log_freq,
            shuffle=True,
            num_workers=6,
            verbose=2, 
            drop_last=True,
        )
    if cfg.mode in ['all', 'eval']:
        r = model.evaluate(val_set, batch_size=cfg.batch_size, num_workers=4)
        print(r)

if __name__ == '__main__':
    import fire
    fire.Fire({"run": run})
