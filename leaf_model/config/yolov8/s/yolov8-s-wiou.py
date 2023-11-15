_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-ag-mixup.py',
]



model = dict(
    bbox_head = dict(
        loss_bbox = dict(
            iou_mode='wiou'
        )
    )
)