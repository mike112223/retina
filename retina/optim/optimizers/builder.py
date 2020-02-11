import torch.optim as torch_optim
from retina.utils import build_from_cfg


def build_optimizer(cfg, default_args=None):
    optimizer = build_from_cfg(cfg, torch_optim, default_args, 'module')
    return optimizer
