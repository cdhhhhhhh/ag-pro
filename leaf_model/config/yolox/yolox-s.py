_base_ = '/home/neau/sdb/mmyolo/configs/yolox/yolox_s_fast_8xb8-300e_coco.py'

num_classes = 1  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 4
# Worker to pre-fetch data for each single GPU during tarining
train_num_workers = 4
# Presistent_workers must be False if num_workers is 0
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.01
max_epochs = 300  # Maximum training epochs

model_test_cfg = dict(
    yolox_style=True,  # better
    # The config of multi-label for multi-class prediction
    multi_label=True,  # 40.5 -> 40.7
    score_thr=0.001,  # Threshold to filter out boxes
    max_per_img=300,  # Max number of detections of each image
    nms=dict(type='nms', iou_threshold=0.65))  # NMS type and threshold

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 4
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 4

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
# generate new random resize shape interval
batch_augments_interval = 10

# -----train val related-----
weight_decay = 0.0005
loss_cls_weight = 1.0
loss_bbox_weight = 5.0
loss_obj_weight = 1.0
loss_bbox_aux_weight = 1.0
center_radius = 2.5  # SimOTAAssigner
num_last_epochs = 15
random_affine_scaling_ratio_range = (0.1, 2)
mixup_ratio_range = (0.8, 1.6)
# Save model checkpoint and validation intervals
save_epoch_intervals = 10
# The maximum checkpoints to keep.
max_keep_ckpts = 3

ema_momentum = 0.0001

# ===============================Unmodified in most cases====================
# model settings
model = dict(
    _delete_ = True,
    type='YOLODetector',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,  # math.sqrt(5)
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    # TODO: Waiting for mmengine support
    use_syncbn=False,
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='YOLOXBatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=batch_augments_interval)
        ]),
    backbone=dict(
        type='YOLOXCSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOXHead',
        head_module=dict(
            type='YOLOXHeadModule',
            num_classes=num_classes,
            in_channels=256,
            feat_channels=256,
            widen_factor=widen_factor,
            stacked_convs=2,
            featmap_strides=(8, 16, 32),
            use_depthwise=False,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='mmdet.IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=loss_bbox_weight),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=loss_obj_weight),
        loss_bbox_aux=dict(
            type='mmdet.L1Loss',
            reduction='sum',
            loss_weight=loss_bbox_aux_weight)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.SimOTAAssigner',
            center_radius=center_radius,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
    test_cfg=model_test_cfg)







train_dataloader = dict(
     batch_size=train_batch_size_per_gpu,
    num_workers=4,
    dataset=dict(
    pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            # dict(
            #     type='mmdet.RandomFlip',
            #     direction = ['horizontal', 'vertical'],
            #     prob = 1.
            #      ),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    # 'flip',
                    # 'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
    ))



train_batch_size_per_gpu = 4










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


data_root = '/home/neau/trainset/crop_leaf'
metainfo = dict(classes=('leaf'))

_base_.train_dataloader.dataset.metainfo = metainfo
_base_.train_dataloader.dataset.data_root = data_root
_base_.train_dataloader.dataset.ann_file = 'result_crop_file_train_class1.json'
_base_.train_dataloader.dataset.data_prefix = dict(img='./')

_base_.val_dataloader.dataset.metainfo = metainfo
_base_.val_dataloader.dataset.data_root = data_root
_base_.val_dataloader.dataset.ann_file = 'result_crop_file_val_class1.json'
_base_.val_dataloader.dataset.data_prefix = dict(img='./')

_base_.test_dataloader = _base_.val_dataloader

_base_.val_evaluator.ann_file = data_root + '/result_crop_file_val_class1.json'
_base_.val_evaluator.proposal_nums = (100, 300, 1000)  # coco设定



_base_.test_evaluator = _base_.val_evaluator




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
    max_epochs=300,
    dynamic_intervals=None,
    val_interval=10
)

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=1
    )
)
