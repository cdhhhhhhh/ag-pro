_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]


model = dict(
    neck=dict(
        type = 'YOLOv8FPAN',
        out_channels=[
            512,
        ],
        use_conv='dcn',
        channel_attention=True,
    ),
     backbone=dict(
        channel_attention=True,
        use_conv='dcn',
        use_msppf=True,
    ),
    bbox_head=dict(
        head_module=dict(
            featmap_strides=[
                8,
            ],
            in_channels=[
                512,
            ],
        ),
        prior_generator=dict(
            strides=[
                8,
            ],
        ),
    ))
