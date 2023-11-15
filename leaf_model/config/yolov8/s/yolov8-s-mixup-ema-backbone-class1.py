_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mixup-class1.py',
]



model = dict(
    backbone = dict(
        attention = 'ema'
    )
)
