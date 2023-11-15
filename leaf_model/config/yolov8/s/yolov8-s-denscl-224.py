_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py',
]

add_config = '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/base.py'

project_name = 'yolov8-s-denscl-lr-f'
train_batch_size_per_gpu = 4



model = dict(
    backbone = dict(
        init_cfg=dict(type='Pretrained', checkpoint='/home/neau/sdb/ag-pro/leaf_model/checkpoint/yolo8s-1.pth'),
    )
)





train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
)


_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
# _base_.optim_wrapper.optimizer.lr = 0.01 / 10
