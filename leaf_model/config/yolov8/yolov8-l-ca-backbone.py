_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l.py',
]



project_name = 'yolov8-l-ca-backbone'




model = dict(
    backbone = dict(
        attention = 'ca'
    )
)
