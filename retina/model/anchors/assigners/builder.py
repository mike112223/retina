from retina.utils import build_from_cfg

from .registry import ASSIGNERS


def build_assigner(cfg, default_args=None):
    assigner = build_from_cfg(cfg, ASSIGNERS, default_args)
    return assigner
