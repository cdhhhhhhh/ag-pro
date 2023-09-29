_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py',
]

# add_config = '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/base.py'

# project_name = 'yolov8-s'
train_batch_size_per_gpu = 2


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers = 4
)


# _base_.default_hooks.param_scheduler.max_epochs = 1000
# _base_.train_cfg.max_epochs = 1000
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

# _base_.model.backbone.norm_cfg.type = 'SyncBN'
# _base_.model.bbox_head.head_module.norm_cfg.type = 'SyncBN'
# _base_.model.neck.norm_cfg.type = 'SyncBN'


val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers = 4
)


load_from = '/home/neau/sdb/ag-pro/leaf_model/checkpoint/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'


custom_imports = dict(
    imports=['leaf_model_tools.hooks','leaf_model_tools.AFPN'],
    allow_failed_imports=False)


custom_hooks = _base_.custom_hooks + [
    dict(type='MySelfExpHook'),
]


data_root = '/home/neau/trainset/crop_leafs_dataset'
metainfo = dict(classes=('round','sharp'))

_base_.train_dataloader.dataset.metainfo = metainfo
_base_.train_dataloader.dataset.data_root = data_root
_base_.train_dataloader.dataset.ann_file = 'train_annotations.json'
_base_.train_dataloader.dataset.data_prefix = dict(img='./')

_base_.val_dataloader.dataset.metainfo = metainfo
_base_.val_dataloader.dataset.data_root = data_root
_base_.val_dataloader.dataset.ann_file = 'val_annotations.json'
_base_.val_dataloader.dataset.data_prefix = dict(img='./')

_base_.test_dataloader = _base_.val_dataloader

_base_.val_evaluator.ann_file = data_root + '/val_annotations.json'
_base_.val_evaluator.proposal_nums = (100, 300, 1000) ## coco设定

_base_.test_evaluator = _base_.val_evaluator





model = dict(
    bbox_head= dict(
        head_module = dict(
            num_classes = 2
        )
    ),
    train_cfg = dict(
        assigner = dict(
            num_classes = 2,
        )
    )
)




default_hooks = dict(
    checkpoint=dict(save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50)
)


visualizer = dict(
    vis_backends= [
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',         
            init_kwargs={
                'project' : 'soybean-leaf',
        })]
    )




