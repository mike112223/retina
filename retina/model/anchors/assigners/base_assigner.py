import torch

from retina.model.utils import bbox_overlaps

from .registry import ASSIGNERS


@ASSIGNERS.register_module
class BaseAssigner(object):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """

        bboxes = bboxes[:, :4]
        overlaps = bbox_overlaps(gt_bboxes, bboxes)
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)

        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.shape

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_tensor([-1] * num_bboxes, dtype=torch.long)
        max_overlaps = overlaps.new_tensor([0.0] * num_bboxes)

        if num_gts == 0:
            assigned_gt_inds[:] = 0

        if gt_labels is None:
            assigned_labels = None
            return num_gts, assigned_gt_inds, assigned_labels, max_overlaps
        else:
            assigned_labels = overlaps.new_tensor([0] * num_bboxes, dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign proposals whose iou with all gts < neg_iou_thr to 0
        assigned_gt_inds[max_overlaps < self.neg_iou_thr] = 0

        # 3. for each bbox, if the iou with its nearest gt >= pos_iou_thr, assign it to that bbox
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. for each gt bbox, assign its nearest proposals (may be more than one) to itself
        for i in range(num_gts):
            if self.gt_max_assign_all:
                assigned_gt_inds[overlaps[i, :] == gt_max_overlaps[i]] = i + 1
            else:
                assigned_gt_inds[gt_argmax_overlaps] = i + 1

        pos_inds = assigned_gt_inds > 0
        assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        return num_gts, assigned_gt_inds, assigned_labels, max_overlaps
