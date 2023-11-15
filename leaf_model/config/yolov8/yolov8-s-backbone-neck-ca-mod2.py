_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]


model = dict(
    backbone=dict(
        channel_attention=True,
        use_conv=dict(
            type='dcn',
            ids=[True, True, True, False],
        ),
        attention=dict(
            type='ema',
            ids=[True, True, True, False],
        ),
        use_msppf=True,
    ),
    neck=dict(
        attention=dict(
            pos='last',
            type='ca',
            ids=[True, True, False]
        )
    )
)
