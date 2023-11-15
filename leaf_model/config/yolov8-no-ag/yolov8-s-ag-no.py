_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s.py',
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
                type='mmdet.RandomFlip',
                # direction = ['horizontal', 'vertical'],
                prob = 0.5
                 ),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    # 'flip',
                    # 'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
    ))



visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs={
                'project': 'soybean-leaf-class2-noag',
            })]
)