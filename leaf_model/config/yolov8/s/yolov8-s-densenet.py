_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov8/yolov8-s.py',
]
model = dict(
    backbone = dict(
        type='mmpretrain.DenseNet'
    )
)