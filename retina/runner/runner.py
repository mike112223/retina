import torch
import os.path as osp
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable

from retina.utils.checkpoint import load_checkpoint, save_checkpoint

from .registry import RUNNERS


@RUNNERS.register_module
class Runner(object):
    """ Runner

    """
    def __init__(self,
                 loader,
                 model,
                 criterion,
                 optim,
                 lr_scheduler,
                 max_epochs,
                 workdir,
                 start_epoch=0,
                 trainval_ratio=1,
                 snapshot_interval=1,
                 gpu=True,
                 test_cfg=None,
                 test_mode=False):
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.metric = None
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.workdir = workdir
        self.trainval_ratio = trainval_ratio
        self.snapshot_interval = snapshot_interval
        self.gpu = gpu
        self.test_cfg = test_cfg
        self.test_mode = test_mode

    def __call__(self):
        if self.test_mode:
            self.test_epoch()
        else:
            assert self.trainval_ratio > 0
            for epoch in range(self.start_epoch, self.max_epochs):
                self.train_epoch()
                self.save_checkpoint(self.workdir)
                if self.trainval_ratio > 0 \
                        and (epoch + 1) % self.trainval_ratio == 0 \
                        and self.loader.get('val'):
                    self.validate_epoch()

    def train_epoch(self):
        print('Epoch %d, Start training' % self.epoch)
        iter_based = hasattr(self.lr_scheduler, '_iter_based')
        for batch in self.loader['train']:

            self.train_batch(batch)
            if iter_based:
                self.lr_scheduler.step()
        if not iter_based:
            self.lr_scheduler.step()

    def validate_epoch(self):
        print('Epoch %d, Start validating' % self.epoch)
        self.metric.reset()
        for img, label in self.loader['val']:
            self.validate_batch(img, label)

    def test_epoch(self):
        print('Start testing')
        print('test info: %s' % self.test_cfg)
        self.metric.reset()
        for img, label in self.loader['val']:
            self.test_batch(img, label)

    def train_batch(self, batch):
        self.model.train()

        self.optim.zero_grad()

        img_metas = batch['img_meta']
        imgs = batch['img']
        gt_bboxes = batch['gt_bboxes']
        gt_labels = batch['gt_labels']
        gt_bboxes_ignore = batch.get('gt_bboxes_ignore', None)

        if self.gpu:
            imgs = imgs.cuda()
            gt_labels = [_.cuda() for _ in gt_labels]
            gt_bboxes = [_.cuda() for _ in gt_bboxes]
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cuda()

        preds_results, targets_results = self.model(
            imgs,
            img_metas,
            self.test_mode,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore
        )

        reg_losses, cls_losses = self.criterion(preds_results, targets_results)
        losses = sum(reg_losses + cls_losses)
        losses.backward()

        self.optim.step()

        if self.iter != 0 and self.iter % 10 == 0:
            print(
                'Train, Epoch %d, Iter %d, LR %s, cls loss %.4f, reg loss %.4f, total loss: %.4f' %
                (self.epoch, self.iter, self.lr, sum(cls_losses).item(),
                 sum(reg_losses).item(), sum(cls_losses).item() + sum(reg_losses).item()))

    def validate_batch(self, img, label):
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()

            pred = self.model(img)

            prob = pred.softmax(dim=1)
            _, pred_label = torch.max(prob, dim=1)
            self.metric.add(pred_label.cpu().numpy(), label.cpu().numpy())
            miou, ious = self.metric.miou()
            print('Validate, mIoU %.4f, IoUs %s' % (miou, ious))

    def test_batch(self, img, label):
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()

            if self.test_cfg:
                scales = self.test_cfg.get('scales', [1.0])
                flip = self.test_cfg.get('flip', False)
                biases = self.test_cfg.get('bias', [0.0])
            else:
                scales = [1.0]
                flip = False
                biases = [0.0]

            assert len(scales) == len(biases)

            n, c, h, w = img.size()
            probs = []
            for scale, bias in zip(scales, biases):
                new_h, new_w = int(h*scale + bias), int(w*scale+bias)
                new_img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=True)
                prob = self.model(new_img).softmax(dim=1)
                probs.append(prob)

                if flip:
                    flip_img = new_img.flip(3)
                    flip_prob = self.model(flip_img).softmax(dim=1)
                    prob = flip_prob.flip(3)
                    probs.append(prob)
            prob = torch.stack(probs, dim=0).mean(dim=0)

            _, pred_label = torch.max(prob, dim=1)
            self.metric.add(pred_label.cpu().numpy(), label.cpu().numpy())
            miou, ious = self.metric.miou()
            print('Test, mIoU %.4f, IoUs %s' % (miou, ious))

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if self.epoch % self.snapshot_interval == 0 or self.epoch == self.max_epochs:
            if meta is None:
                meta = dict(epoch=self.epoch, iter=self.iter, lr=self.lr)
            else:
                meta.update(epoch=self.epoch, iter=self.iter, lr=self.lr)

            filename = filename_tmpl.format(self.epoch)
            filepath = osp.join(out_dir, filename)
            linkpath = osp.join(out_dir, 'latest.pth')
            optimizer = self.optim if save_optimizer else None
            print('Save checkpoint %s', filename)
            save_checkpoint(self.model,
                            filepath,
                            optimizer=optimizer,
                            meta=meta)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        print('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict)

    @property
    def epoch(self):
        """int: Current epoch."""
        return self.lr_scheduler.last_epoch

    @epoch.setter
    def epoch(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_epoch = val

    @property
    def lr(self):
        lr = [x['lr'] for x in self.optim.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optim.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    @property
    def iter(self):
        """int: Current iteration."""
        return self.lr_scheduler.last_iter

    @iter.setter
    def iter(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_iter = val

    def resume(self,
               checkpoint,
               resume_optimizer=False,
               resume_lr=True,
               resume_epoch=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim.load_state_dict(checkpoint['optimizer'])
        if resume_epoch:
            self.epoch = checkpoint['meta']['epoch']
            self.start_epoch = self.epoch
            self.iter = checkpoint['meta']['iter']
        if resume_lr:
            self.lr = checkpoint['meta']['lr']