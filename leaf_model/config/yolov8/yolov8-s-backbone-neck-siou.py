_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]


model = dict(
    backbone=dict(
        channel_attention=True,
        use_conv='dcn',
        use_msppf=True,
    ),
    bbox_head=dict(
        loss_bbox=dict(
            iou_mode='siou'
        )
    )
)
