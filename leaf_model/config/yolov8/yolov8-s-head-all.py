_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s.py',
]



model = dict(
    bbox_head = dict(
        head_module = dict(
            use_attention = 'all'
        )
    )
)


