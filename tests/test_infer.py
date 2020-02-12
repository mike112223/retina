import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.abspath('../retina'))


def test_infer():

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

    for batch in train_loader:
        img_metas = batch['img_meta']
        # print([_['filename'] for _ in img_metas])
        imgs = batch['img']
        gt_bboxes = batch['gt_bboxes']
        gt_labels = batch['gt_labels']
        gt_bboxes_ignore = batch.get('gt_bboxes_ignore', None)

        bbox_results = model(
            imgs,
            img_metas,
            True,
            num_classes=cfg['num_classes']
        )

if __name__ == '__main__':
    test_infer()
