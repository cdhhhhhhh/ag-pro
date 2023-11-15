_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]

img_scale=(1280, 1280)

train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Mosaic',
                img_scale=img_scale,
                pad_val=114.0,
                pre_transform=[
                    dict(type='LoadImageFromFile',
                         backend_args=_base_.backend_args),
                    dict(type='LoadAnnotations', with_bbox=True)
                ]
            ),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(1 - 0.5, 1 + 0.5),
                max_aspect_ratio=100,
                # img_scale is (width, height)
                border=(-img_scale[0] // 2, -img_scale[0] // 2),
                border_val=(114, 114, 114)),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
)




model = dict(
    backbone=dict(
        out_indices=(1, 2, 3, 4),
        channel_attention=True,
        use_conv='dcn',
        use_msppf=True,   
    ),
    neck=dict(
        attention = dict(
             pos = 'last',
             type = 'ca'
         ),
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        out_channels=[
            128,
            256,
            512,
            1024,
        ],
    ),
    bbox_head=dict(
        prior_generator=dict(
            strides=[
                4,
                8,
                16,
                32,
            ]),
        head_module=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            in_channels=[
                128,
                256,
                512,
                1024,
            ],
        ),
    )
)




train_batch_size_per_gpu = 1


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4
)
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=1
)


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]




val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline))


