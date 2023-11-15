_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py',
]

train_batch_size_per_gpu = 4


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4
)

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu



val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4
)


load_from = None

custom_imports = dict(
    imports=['leaf_model_tools.hooks',
             'leaf_model_tools.GDNeck',
             'leaf_model_tools.AFPN', 
             'leaf_model_tools.AFPN4', 
             'leaf_model_tools.EMA'
             ],
    allow_failed_imports=False)

custom_hooks = _base_.custom_hooks + [
    dict(type='MySelfExpHook'),
]

data_root = '/home/neau/trainset/leaf-o'

# data_root = '/home/neau/trainset/crop_leaf'
metainfo = dict(classes=('leaf'))

_base_.train_dataloader.dataset.metainfo = metainfo
_base_.train_dataloader.dataset.data_root = data_root
_base_.train_dataloader.dataset.ann_file = 'result_crop_file_train.json'
_base_.train_dataloader.dataset.data_prefix = dict(img='./')

_base_.val_dataloader.dataset.metainfo = metainfo
_base_.val_dataloader.dataset.data_root = data_root
_base_.val_dataloader.dataset.ann_file = 'result_crop_file_val.json'
_base_.val_dataloader.dataset.data_prefix = dict(img='./')

_base_.test_dataloader = _base_.val_dataloader

_base_.val_evaluator.ann_file = data_root + '/result_crop_file_val.json'
_base_.val_evaluator.proposal_nums = (100, 300, 1000)  # coco设定
_base_.val_evaluator.classwise = True



_base_.test_evaluator = _base_.val_evaluator


model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=1
        )
    ),
    train_cfg=dict(
        assigner=dict(
            num_classes=1,
        )
    )
)


default_hooks = dict(
    checkpoint=dict(save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50)
)


visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs={
                'project': 'soybean-leaf-class1-noag',
            })]
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=200,
    dynamic_intervals=None,
    val_interval=10
)

default_hooks = dict(
    param_scheduler=dict(
        max_epochs=200
    ),
    checkpoint=dict(
        max_keep_ckpts=1
    )
)
