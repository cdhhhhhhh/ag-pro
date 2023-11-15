_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mosaic9.py',
]



model = dict(
       backbone = dict(
        attention = 'ema'
    ),
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
        backbone = dict(
        attention = 'ema'
    )
)


model = dict(
 
    neck = dict(
        attention = dict(
            type = 'ema',
            pos = 'last'
        )
    )
)
