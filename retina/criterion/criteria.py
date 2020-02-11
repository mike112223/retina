import torch.nn as nn
import torch.nn.functional as F

from .builder import build_loss

from retina.model.utils import multi_apply


class Criteria(nn.Module):
    def __init__(self, cls_loss_cfg, reg_loss_cfg, num_classes):
        super(Criteria, self).__init__()
        self.cls_loss = build_loss(cls_loss_cfg)
        self.reg_loss = build_loss(reg_loss_cfg)
        self.num_classes = num_classes

    def forward(self, preds_results, targets_results):

        labels, label_weights, bbox_targets, bbox_weights, num_total_pos, num_total_neg = targets_results

        cls_scores, bbox_preds = preds_results

        reg_losses, cls_losses = multi_apply(
            self.forward_single,
            bbox_preds,
            bbox_targets,
            bbox_weights,
            cls_scores,
            labels,
            label_weights,
            num_total_pos=num_total_pos,
        )

        # losses: (list of levels)
        return reg_losses, cls_losses

    def forward_single(self, 
                       bbox_pred,
                       bbox_target,
                       bbox_weight,
                       cls_score,
                       label,
                       label_weight,
                       num_total_pos):
        # reg loss
        bbox_target = bbox_target.reshape(-1, 4)
        bbox_weight = bbox_weight.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        reg_loss = self.reg_loss(bbox_pred, bbox_target, bbox_weight, num_total_pos)

        # cls loss
        label = label.reshape(-1)
        label = F.one_hot(label, self.num_classes+1)[:, 1:]
        label_weight = label_weight.reshape(-1)
        label_weight = label_weight.view(-1, 1).expand(
            label_weight.size(0), self.num_classes)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        cls_loss = self.cls_loss(cls_score, label, label_weight, num_total_pos)

        return reg_loss, cls_loss
