_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/class1/yolov8-s-ag-no.py',
]


model = dict(
    neck=dict(
        use_conv='dw_inception',
        channel_attention=True,
    ),
    backbone=dict(
        channel_attention=True,
        use_conv='dcn',
        use_msppf=True,
    ),
)
