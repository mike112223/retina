from retina.utils import build_from_cfg

from .registry import DETECTORS


def build_detector(cfg, default_args=None):
    detector = build_from_cfg(cfg, DETECTORS, default_args)
    return detector
