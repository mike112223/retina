
workdir = 'workdir'

img_norm_cfg = dict(mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375))
ignore_label = 255

dataset_type = 'VOCDataset'
dataset_root = 'data/voc/'

data = dict(
    train=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=[
                dataset_root + 'VOC2007_trainval/ImageSets/Main/trainval.txt',
                dataset_root + 'VOC2012_trainval/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[
                dataset_root + 'VOC2007_trainval/',
                dataset_root + 'VOC2012_trainval/'
            ]
        ),
        transforms=[
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(1344, 800)),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ],
        loader=dict(
            type='DataLoader',
            batch_size=2,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        ),
    )
)

octave_base_scale = 4
scales_per_octave = 3
anchor_ratios = [0.5, 1.0, 2.0]
num_classes = 20

model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        arch='resnet50',
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5
    ),
    head=dict(
        type='RetinaHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=octave_base_scale,
        scales_per_octave=scales_per_octave,
        anchor_ratios=anchor_ratios,
    ),
    anchor=dict(
        type='Anchor',
        octave_base_scale=octave_base_scale,
        scales_per_octave=scales_per_octave,
        anchor_ratios=anchor_ratios,
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        allowed_border=-1,
        assigner=dict(
            type='BaseAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=.0,
        ),
        sampler=dict(
            type='BaseSampler',
        )

    )
)

criterion = dict(
    cls_loss=dict(
        type='FocalLoss',
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0
    ),
    reg_loss=dict(
        type='SmoothL1Loss',
        beta=0.11,
        loss_weight=1.0
    )
)

optim = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001
    ),
    lr_scheduler=dict(
        type='StepLR',
        lr_step=[8, 11],
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
    )
)

runner = dict(
    type='Runner',
    max_epochs=12,
    trainval_ratio=1,
    snapshot_interval=5,
)
