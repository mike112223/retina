import os
import sys

import cv2

sys.path.insert(0, os.path.abspath('../retina'))


def test_datasets():

    from retina.data.datasets import build_dataset
    from retina.data.datasets.transforms import build_transform
    from retina.data.dataloaders import build_dataloader, default_collate

    dataset_type = 'VOCDataset'
    dataset_root = 'data/voc/'

    dataset = dict(
        type=dataset_type,
        ann_file=dataset_root + 'VOC2007_trainval/ImageSets/Main/val.txt',
        img_prefix=dataset_root + 'VOC2007_trainval/',
    )

    transforms = [
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Pad', size=(1344, 800)),
        dict(
            type='Collect',
            keys=['img', 'gt_bboxes', 'gt_labels'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                       'scale_factor', 'flip')
        ),
    ]

    loader = dict(
        type='DataLoader',
        batch_size=1,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )

    train_tf = build_transform(transforms)
    train_dataset = build_dataset(dataset, dict(transforms=train_tf))

    train_loader = build_dataloader(loader, dict(dataset=train_dataset, collate_fn=default_collate))

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, batch in enumerate(train_loader):

        img = batch['img'].numpy()[0].transpose(1, 2, 0)
        gt_bboxes = batch['gt_bboxes'][0].numpy()
        gt_labels = batch['gt_labels'][0].numpy()
        img_meta = batch['img_meta'][0]

        for j, gt_bbox in enumerate(gt_bboxes):
            img = cv2.rectangle(
                img,
                (gt_bbox[0], gt_bbox[1]),
                (gt_bbox[2], gt_bbox[3]),
                (0, 255, 0),
                2
            )

            img = cv2.putText(
                img,
                '%d' % gt_labels[j],
                (gt_bbox[0], gt_bbox[1]),
                font,
                1.2,
                (255, 0, 0),
                2
            )

        save_path = os.path.join('tests/test_imgs/', img_meta['filename'].split('/')[-1])

        cv2.imwrite(save_path, img)

        with open(save_path[:-3] + 'txt', 'w') as f:
            f.write(str(img_meta))

        if i == 30:
            break

if __name__ == '__main__':
    test_datasets()
