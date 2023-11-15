_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s.py',
]


model = dict(
    backbone=dict(
        arch='P5',
        type='YOLOv8CSPDarknet',
        out_indices = (1, 2, 3, 4),
    ),
    neck = dict(
        in_channels=[
            128,
            256,
            512,
            1024
        ],
        out_channels=[
            128,
            256,
            512,
            1024
        ],
    ),
 
    bbox_head = dict(
        prior_generator=dict(
        strides=[
            4,
            8,
            16,
            32
        ]),
        head_module = dict(
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





add_config = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/base.py',
]
