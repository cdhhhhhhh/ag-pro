_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]



model = dict(
    backbone = dict(
        use_msppf = True
    )
)



resume = True