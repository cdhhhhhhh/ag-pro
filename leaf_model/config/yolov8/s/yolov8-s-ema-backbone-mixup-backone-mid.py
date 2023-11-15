_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-ag-mixup.py',
]



model = dict(
    backbone = dict(
        attention = 'ema'
    ),
    neck = dict(
        attention = dict(
            type = 'ema',
            pos = 'mid'
        )
    )
)
