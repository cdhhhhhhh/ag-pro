_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py',
]

add_config = '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/base.py'

project_name = 'yolov8-l-denscl-224-1.5-lr-0.1-SyncBN'
train_batch_size_per_gpu = 2



train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
)

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

# find_unused_parameters=True

_base_.optim_wrapper.optimizer.lr = 0.1

_base_.model.backbone.norm_cfg.type = 'SyncBN'
_base_.model.bbox_head.head_module.norm_cfg.type = 'SyncBN'
_base_.model.neck.norm_cfg.type = 'SyncBN'



model = dict(
    backbone = dict(
        init_cfg=dict(type='Pretrained', checkpoint='/home/neau/sdb/ag-pro/leaf_model/checkpoint/csp_224_1.5.pth'),
    )
)

