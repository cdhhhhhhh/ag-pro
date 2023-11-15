_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]


model = dict(
    backbone=dict(
        arch='P5-m',
        last_stage_out_channels=64
    ),
    neck=dict(
        type='YOLOv8FPAN',
        in_channels=[
            256,
            128,
            64
        ],
        out_channels=[
            256,
        ],
    ),
    bbox_head=dict(
        head_module=dict(
            featmap_strides=[
                8,
            ],
            in_channels=[
                256,
            ],
        ),
        prior_generator=dict(
            strides=[
                8,
            ],
        ),
    )
)
