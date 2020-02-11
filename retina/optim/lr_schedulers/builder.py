from retina.utils import build_from_cfg

from .registry import LR_SCHEDULERS


def build_lr_scheduler(cfg, default_args=None):
    lr_scheduler = build_from_cfg(cfg, LR_SCHEDULERS, default_args)
    return lr_scheduler
