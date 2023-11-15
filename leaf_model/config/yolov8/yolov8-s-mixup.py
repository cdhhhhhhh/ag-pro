_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s.py',
]


train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        img_scale=(
                            640,
                            640,
                        ),
                        pad_val=114.0,
                        pre_transform=[
                            dict(backend_args=None, type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', with_bbox=True),
                        ],
                        type='Mosaic'),
                    dict(
                        border=(
                            -320,
                            -320,
                        ),
                        border_val=(
                            114,
                            114,
                            114,
                        ),
                        max_aspect_ratio=100,
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.5,
                            2,
                        ),
                        type='YOLOv5RandomAffine'),
                ],
                prob=1,
                type='YOLOv5MixUp'),
            # dict(
            #     type='YOLOv5RandomAffine',
            #     max_rotate_degree=0.0,
            #     max_shear_degree=0.0,
            #     scaling_ratio_range=(1 - 0.5, 1 + 0.5),
            #     max_aspect_ratio=100,
            #     # img_scale is (width, height)
            #     border=(-640 // 2, -640 // 2),
            #     border_val=(114, 114, 114)),
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

# resume = True