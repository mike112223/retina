import torch.nn as nn

from .registry import DETECTORS

from retina.model.backbones import build_backbone
from retina.model.necks import build_neck
from retina.model.heads import build_head
from retina.model.anchors import build_anchor
from retina.model.utils import bbox2result


@DETECTORS.register_module
class RetinaNet(nn.Module):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck,
                 head,
                 anchor,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__()

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.anchor = build_anchor(anchor)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)

        # cls_scores (list of levels, default len is 5 (p3-p7))
        # cls_score (each level, shape: [batch(imgs), num_classes * anchors, h, w])
        # similarly， bbox_preds
        cls_scores, bbox_preds = self.head(x)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        # anchor_list (list of list, [img[level, ...], ...])
        anchor_list, valid_flag_list = self.anchor.get_anchors(
            featmap_sizes,
            img_metas,
            device=cls_scores[0].device
        )

        # labels, label_weights, bbox_targets, bbox_weights, num_total_pos, num_total_neg
        targets_results = self.anchor.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore,
            gt_labels
        )

        pred_results = (cls_scores, bbox_preds)

        return pred_results, targets_results

    def forward_test(self, img, img_metas, num_classes):
        x = self.extract_feat(img)

        # cls_scores (list of levels, default len is 5 (p3-p7))
        # cls_score (each level, shape: [batch(imgs), num_classes * anchors, h, w])
        # similarly， bbox_preds
        cls_scores, bbox_preds = self.head(x)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        # anchor_list (list of list, [img[level, ...], ...])
        anchor_list, valid_flag_list = self.anchor.get_anchors(
            featmap_sizes,
            img_metas,
            device=cls_scores[0].device
        )

        bbox_list = self.anchor.get_bboxes(
            anchor_list,
            valid_flag_list,
            img_metas,
            cls_scores,
            bbox_preds,
            num_classes
        )

        bbox_results = [
            bbox2result(det_bboxes, det_labels, num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        return bbox_results

    def forward(self, img, img_meta, test_mode=False, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if test_mode:
            return self.forward_test(img, img_meta, **kwargs)
        else:
            return self.forward_train(img, img_meta, **kwargs)
