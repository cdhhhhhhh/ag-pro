_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-denscl-224.py',
]



project_name = 'yolov8-s-denscl-lr-0.03'
_base_.optim_wrapper.optimizer.lr = 0.03

