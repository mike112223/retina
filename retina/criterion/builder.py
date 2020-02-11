from retina.utils import build_from_cfg

from .registry import LOSSES


def build_loss(cfg, default_args=None):
    loss = build_from_cfg(cfg, LOSSES, default_args)
    return loss
