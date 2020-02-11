from retina.utils import build_from_cfg

from .registry import NECKS


def build_neck(cfg, default_args=None):
    neck = build_from_cfg(cfg, NECKS, default_args)
    return neck
