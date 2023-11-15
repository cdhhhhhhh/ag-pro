_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]


model = dict(
    backbone=dict(
        out_indices=(1, 2, 3, 4),
    ),
    neck=dict(
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        out_channels=[
            128,
        ],
    ),
    bbox_head=dict(
        prior_generator=dict(
            strides=[
                4,
            ]),
        head_module=dict(
            featmap_strides=[
                4,
            ],
            in_channels=[
                128,
            ],
        ),
    )
)
