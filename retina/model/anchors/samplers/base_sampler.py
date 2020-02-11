
import torch

# from .sampling_result import SamplingResult

from .registry import SAMPLERS


@SAMPLERS.register_module
class BaseSampler(object):
    def __init__(self):
        pass

    def sample(self, assign_result, bboxes, gt_bboxes):
        num_gts, assigned_gt_inds, assigned_labels, max_overlaps = assign_result

        pos_inds = torch.nonzero(
            assigned_gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assigned_gt_inds == 0).squeeze(-1).unique()
        # gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        # sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
        #                                  assign_result, gt_flags)
        return pos_inds, neg_inds
