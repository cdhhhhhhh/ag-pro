_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-denscl-224.py',
]

project_name = 'yolov8-s-denscl-frozen-1'


model = dict(
    backbone = dict(
        frozen_stages = 1
    )
)
find_unused_parameters=True