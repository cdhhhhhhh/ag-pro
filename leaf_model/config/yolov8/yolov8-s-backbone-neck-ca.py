_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]


model = dict(
    backbone=dict(
        channel_attention=False,
        use_conv=dict(
            type='dcn',
            ids=[False, True, True, True],
        ),
        attention=dict(
            type='ema',
            ids=[False, True, True, False],
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
