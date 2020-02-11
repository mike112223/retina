import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.abspath('../retina'))


def test_config_build_all_modules():

    from retina.utils import Config
    from retina.data.datasets import build_dataset
    from retina.data.datasets.transforms import build_transform
    from retina.data.dataloaders import build_dataloader, default_collate
    from retina.model import build_detector
    from retina.criterion import Criteria
    from retina.optim.optimizers import build_optimizer
    from retina.optim.lr_schedulers import build_lr_scheduler

    cfg_fp = os.path.join(os.path.abspath('configs'), 'test.py')
    cfg = Config.fromfile(cfg_fp)

    train_tf = build_transform(cfg['data']['train']['transforms'])
    train_dataset = build_dataset(cfg['data']['train']['dataset'], dict(transforms=train_tf))

    train_loader = build_dataloader(cfg['data']['train']['loader'], dict(dataset=train_dataset, collate_fn=default_collate))

    model = build_detector(cfg['model'])

    criterion = Criteria(
        cls_loss_cfg=cfg['criterion']['cls_loss'],
        reg_loss_cfg=cfg['criterion']['reg_loss'],
        num_classes=cfg['num_classes']
    )

    optimizer = build_optimizer(
        cfg['optim']['optimizer'],
        dict(params=model.parameters())
    )
    lr_scheduler = build_lr_scheduler(
        cfg['optim']['lr_scheduler'],
        dict(optimizer=optimizer, niter_per_epoch=len(train_loader))
    )

if __name__ == '__main__':
    test_config_build_all_modules()
