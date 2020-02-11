from retina.utils import build_from_cfg

from .registry import ANCHORS


def build_anchor(cfg, default_args=None):
    anchor = build_from_cfg(cfg, ANCHORS, default_args)
    return anchor
