import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath('../retina'))

def _demo_mm_inputs(
        input_shape=(1, 3, 300, 300), num_items=None, num_classes=10):
    """
    Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_items (None | List[int]):
            specifies the number of boxes in each batch item

        num_classes (int):
            number of different labels a box might have
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]

    gt_bboxes = []
    gt_labels = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
        tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
        br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
        br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
    }
    return mm_inputs

def test_retina_forward():

    from retina.utils import Config
    from retina.model import build_detector
    from retina.criterion import Criteria

    # init
    cfg_fp = os.path.join(os.path.abspath('configs'), 'test.py')
    cfg = Config.fromfile(cfg_fp)

    model = build_detector(cfg['model'])

    criterion = Criteria(
        cls_loss_cfg=cfg['criterion']['cls_loss'],
        reg_loss_cfg=cfg['criterion']['reg_loss'],
        num_classes=cfg['num_classes']
    )

    # input
    input_shape = (3, 3, 224, 224)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']

    # forward
    preds_results, targets_results = model(
        imgs,
        img_metas,
        False,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )

    reg_losses, cls_losses = criterion(preds_results, targets_results)
    print(reg_losses, cls_losses)

    # cuda forward
    if torch.cuda.is_available():
        model = model.cuda()
        imgs = imgs.cuda()
        # Test forward train
        gt_bboxes = [b.cuda() for b in mm_inputs['gt_bboxes']]
        gt_labels = [g.cuda() for g in mm_inputs['gt_labels']]

        preds_results, targets_results = model(
            imgs,
            img_metas,
            False,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
        )

        reg_losses, cls_losses = criterion(preds_results, targets_results)
        print(reg_losses, cls_losses)


if __name__ == '__main__':
    test_retina_forward()
