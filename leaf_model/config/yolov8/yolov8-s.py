_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py',
]

add_config = '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/base.py'

# project_name = 'yolov8-s'
train_batch_size_per_gpu = 4


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
)


# _base_.default_hooks.param_scheduler.max_epochs = 1000
# _base_.train_cfg.max_epochs = 1000
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

# _base_.model.backbone.norm_cfg.type = 'SyncBN'
# _base_.model.bbox_head.head_module.norm_cfg.type = 'SyncBN'
# _base_.model.neck.norm_cfg.type = 'SyncBN'
