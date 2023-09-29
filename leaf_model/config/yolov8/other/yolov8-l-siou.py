_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l.py',
]


project_name = 'yolov8-l-siou'


model = dict(
    bbox_head = dict(
        loss_bbox = dict(
            iou_mode='siou'
        )
    )
)

