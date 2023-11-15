_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]
norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)

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
        type='BiFPN',
        num_stages=6,
        in_channels=[256, 512, 1024],
        out_channels=160,
        start_level=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)),
)
