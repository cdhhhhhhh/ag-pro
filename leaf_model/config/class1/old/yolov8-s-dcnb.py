_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/class1/yolov8-s-ag-no.py',
]


model = dict(
    backbone=dict(
        channel_attention=True,
        use_conv='dcn',
    )
)
