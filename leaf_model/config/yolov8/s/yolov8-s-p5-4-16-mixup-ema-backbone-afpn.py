_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-p5-4-16-mixup.py',
]


model = dict(
    neck=dict(
    _delete_ = True,
      in_channels=[
            128,
            256,
            512,
        ],
        out_channels=[
            128,
            256,
            512,
        ],
    type='YOLOv8AFPN',
    widen_factor=0.5)
)






