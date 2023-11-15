_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-ag-mixup.py',
]


model = dict(
    backbone=dict(
        arch='P4',
        type='YOLOv8CSPDarknet',
        out_indices = (1, 2, 3),
        last_stage_out_channels=512,
    ),
    neck = dict(
        in_channels=[
            128,
            256,
            512,
        ],
        out_channels=[
            128,
            256,
            512,
        ],
    ),
    bbox_head = dict(
        prior_generator=dict(
        strides=[
            4,
            8,
            16,
        ]),
        head_module = dict(
            featmap_strides=[
                4,      
                8,
                16,
            ],
            in_channels=[
                128,
                256,
                512,
            ],
        ),
    )
)






