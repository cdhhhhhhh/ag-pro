_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l.py',
]


project_name = 'yolov8-l-p6-1024'

add_config = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/base.py',
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/base-p6.py',
    # '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/base-1024.py'
]
train_batch_size_per_gpu = 1



train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
)

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

