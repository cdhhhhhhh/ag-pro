_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]


model = dict(
    backbone=dict(
        spp_kernel_sizes=3
    ),
    neck=dict(
        type='YOLOv8FPAN',
        out_channels=[
            128,
        ],
    ),
    bbox_head=dict(
        head_module=dict(
            featmap_strides=[
                4,
            ],
            in_channels=[
                128,
            ],
        ),
        prior_generator=dict(
            strides=[
                4,
            ],
        ),
    ))