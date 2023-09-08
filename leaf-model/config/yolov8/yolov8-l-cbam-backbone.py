_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py',
]

add_config = '/home/neau/sdb/ag-pro/leaf-model/config/yolov8/base.py'

project_name = 'yolov8-l-cbam-backbone'
train_batch_size_per_gpu = 2



train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
)

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
