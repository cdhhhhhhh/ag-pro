_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mixup-class1.py',
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
    bbox_head = dict(
        loss_bbox = dict(
            iou_mode='wiou'
        )
    ),
        test_cfg = dict(
        nms=dict(type='soft_nms'),
    )
)

