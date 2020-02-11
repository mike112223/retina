import cv2

import numpy as np

from torch.utils.data import Dataset

from .registry import DATASETS


@DATASETS.register_module
class BaseDataset(Dataset):
    """
    out: results: {
        'img_meta':...,
        'img': (dtype:torch.float32, size:[3, h, w]),
        'gt_bboxes': (dtype:torch.float32, size:[k, 4]'
        'gt_labels: (dtype:torch.int64, size:[k])'
        }
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 transforms=None,
                 test_mode=False):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.transforms = transforms
        self.test_mode = test_mode

        # load annotations
        self.img_infos = self.load_annotations(self.ann_file)
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        pass

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def __getitem__(self, idx):

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        results = self.process(results)

        return results

    def process(self, results):

        # load img
        filename = results['img_info']['filename']
        img = cv2.imread(filename)
        results['img'] = img
        results['filename'] = filename
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        ann_info = results['ann_info']
        # load bbox
        results['bbox_fields'] = []
        results['gt_bboxes'] = ann_info['bboxes']

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        # load label
        results['gt_labels'] = ann_info['labels']

        if self.transforms:
            results = self.transforms(results)

        return results
