_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s.py',
]




model = dict(
    test_cfg = dict(
        nms=dict(type='soft-nms'),
    )
)



