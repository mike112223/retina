
import numpy as np
import torch
import torch.nn as nn

from .assigners import build_assigner
from .samplers import build_sampler

from .anchor_generator import AnchorGenerator
from .registry import ANCHORS

from retina.model.utils import bbox2delta, multi_apply, unmap


@ANCHORS.register_module
class Anchor(object):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories including the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 anchor_scales=None,
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 allowed_border=0,
                 assigner=None,
                 sampler=None):
        super(Anchor, self).__init__()
        if anchor_scales is None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            anchor_scales = octave_scales * octave_base_scale

        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.allowed_border = allowed_border

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        self.assigner = build_assigner(assigner)
        self.sampler = build_sampler(sampler)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None):

        num_imgs = len(img_metas)
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        # get target of each img
        target_results = multi_apply(
            self.get_targets_single,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
        )

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list) = target_results

        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)

        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)


    def get_targets_single(self,
                           flat_anchors,
                           valid_flags,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta):
        # get targets of each img
        inside_flags = self.anchor_inside_flags(flat_anchors, valid_flags,
                                                img_meta['img_shape'][:2])
        if not inside_flags.any():
            return (None, ) * 6

        anchors = flat_anchors[inside_flags, :]

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        # assign_result: num_gts, assigned_gt_inds, assigned_labels, max_overlaps
        assign_result = self.assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_gts, assigned_gt_inds, assigned_labels, max_overlaps = assign_result
        pos_inds, neg_inds = sampling_result

        if len(pos_inds) > 0:
            pos_anchors = anchors[pos_inds]
            gt_rt_pos_anchors = gt_bboxes[assigned_gt_inds[pos_inds] - 1]

            pos_bbox_targets = bbox2delta(
                pos_anchors,
                gt_rt_pos_anchors,
                self.target_means,
                self.target_stds,
            )

            bbox_targets[pos_inds] = pos_bbox_targets
            bbox_weights[pos_inds] = 1.0
            label_weights[pos_inds] = 1.0

        if len(neg_inds):
            label_weights[neg_inds] = 1.0

        labels = assigned_labels
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)


    def anchor_inside_flags(self,
                            flat_anchors,
                            valid_flags,
                            img_shape):
        img_h, img_w = img_shape[:2]
        if self.allowed_border >= 0:
            inside_flags = valid_flags & \
                (flat_anchors[:, 0] >= -self.allowed_border).type(torch.uint8) & \
                (flat_anchors[:, 1] >= -self.allowed_border).type(torch.uint8) & \
                (flat_anchors[:, 2] < img_w + self.allowed_border).type(torch.uint8) & \
                (flat_anchors[:, 3] < img_h + self.allowed_border).type(torch.uint8)
        else:
            inside_flags = valid_flags
        return inside_flags


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets
