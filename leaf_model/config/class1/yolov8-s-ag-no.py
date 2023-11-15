_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/class1/yolov8-s-class1.py',
]


train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='RandomBrightnessContrast', p=0.5),
                    dict(type='GaussianBlur', p=0.5),
                    dict(type='ToGray', p=0.5),
                    dict(type='GaussNoise', p=0.5)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
                keymap={
                    'img': 'image',
                    'gt_bboxes': 'bboxes'
                }),
            dict(
                type='mmdet.RandomFlip',
                direction=['horizontal', 'vertical'],
                prob=0.5
            ),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
    ))



train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    dynamic_intervals=None,
    val_interval=1
)


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))



default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=-1
    ),
    param_scheduler=None
)
