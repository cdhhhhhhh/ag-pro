_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-ag-mixup.py',
]


model = dict(
    backbone=dict(
        arch='P5',
        type='YOLOv8CSPDarknet',
        out_indices=(1, 2, 3, 4),
        attention = 'ema',
    ),
    neck=dict(
        _delete_=True,
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
        attention = True,
        type='YOLOv8AFPN4',
        widen_factor=0.5),
    bbox_head=dict(
        prior_generator=dict(
            strides=[
                4,
                8,
                16,
                32
            ]),
        head_module=dict(
            featmap_strides=[
                4,
                8,
                16,
                32
            ],
            in_channels=[
                128,
                256,
                512,
                1024
            ],
        ),
    )
)
# model = dict(
#     test_cfg = dict(
#         nms=dict(type='soft_nms'),
#     )
# )
