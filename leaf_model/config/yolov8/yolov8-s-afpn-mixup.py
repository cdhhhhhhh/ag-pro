_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-ag-mixup.py',
]



model = dict(
    neck=dict(
        _delete_ = True,
        in_channels=[
            256,
            512,
            1024,
        ],
        out_channels=[
            256,
            512,
            1024,
        ],
        type='YOLOv8AFPN',
        widen_factor=0.5),
)