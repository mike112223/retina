import torch

import numpy as np


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)

    x1s = torch.max(bboxes1[:, None, 0], bboxes2[:, 0])
    y1s = torch.max(bboxes1[:, None, 1], bboxes2[:, 1])
    x2s = torch.min(bboxes1[:, None, 2], bboxes2[:, 2])
    y2s = torch.min(bboxes1[:, None, 3], bboxes2[:, 3])

    ws = torch.max(x2s.new_tensor(0.0), x2s - x1s + 1)
    hs = torch.max(y2s.new_tensor(0.0), y2s - y1s + 1)

    intersections = ws * hs

    if mode == 'iou':
        ious = intersections / (areas1[:, None] + areas2 - intersections)
    else:
        ious = intersections / (areas1[:, None])

    return ious


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):

    pxs = (proposals[:, 0] + proposals[:, 2]) / 2
    pys = (proposals[:, 1] + proposals[:, 3]) / 2
    pws = (proposals[:, 2] - proposals[:, 0]) + 1
    phs = (proposals[:, 3] - proposals[:, 1]) + 1

    gxs = (gt[:, 0] + gt[:, 2]) / 2
    gys = (gt[:, 1] + gt[:, 3]) / 2
    gws = (gt[:, 2] - gt[:, 0]) + 1
    ghs = (gt[:, 3] - gt[:, 1]) + 1

    txs = (gxs - pxs) / pws
    tys = (gys - pys) / phs
    tws = torch.log(gws / pws)
    ths = torch.log(ghs / phs)

    deltas = torch.stack([txs, tys, tws, ths], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)

    deltas = (deltas - means) / stds

    return deltas


def delta2bbox(proposals, deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1], max_shape=None):

    means = deltas.new_tensor(means)
    stds = deltas.new_tensor(stds)
    deltas = deltas * stds + means

    dxs = deltas[:, 0]
    dys = deltas[:, 1]
    dws = deltas[:, 2]
    dhs = deltas[:, 3]

    # Compute center of each roi
    pxs = ((proposals[:, 0] + proposals[:, 2]) * 0.5)
    pys = ((proposals[:, 1] + proposals[:, 3]) * 0.5)
    # Compute width/height of each roi
    pws = (proposals[:, 2] - proposals[:, 0] + 1.0)
    phs = (proposals[:, 3] - proposals[:, 1] + 1.0)
    # Use exp(network energy) to enlarge/shrink each roi
    gws = pws * dws.exp()
    ghs = phs * dhs.exp()
    # Use network energy to shift the center of each roi
    gxs = torch.addcmul(pxs, 1, pws, dxs)  # gxs = pxs + pws * dxs
    gys = torch.addcmul(pys, 1, phs, dys)  # gys = pys + phs * dys
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gxs - gws * 0.5 + 0.5
    y1 = gys - ghs * 0.5 + 0.5
    x2 = gxs + gws * 0.5 - 0.5
    y2 = gys + ghs * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)

    return bboxes


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]
