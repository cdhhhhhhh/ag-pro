_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s.py',
]



project_name = 'yolov8-s-ca-neck'



model = dict(
    neck = dict(
        attention = 'ca'
    )
)
