default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.1,
        max_epochs=300),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_param_scheduler=False,
        save_best='auto',
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='soybean-leaf',
                name='yolov7_x_syncbn_fast_8x16b-300e_coco',
                notes='amp'))
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_x_syncbn_fast_8x16b-300e_coco/yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth'
resume = False
file_client_args = dict(backend='disk')
_file_client_args = dict(backend='disk')
tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))
img_scales = [(640, 640), (320, 320), (960, 960)]
_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(320, 320)),
            dict(
                type='LetterResize',
                scale=(320, 320),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
            dict(
                type='LetterResize',
                scale=(960, 960),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ])
]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (640, 640)
            }, {
                'type': 'LetterResize',
                'scale': (640, 640),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (320, 320)
            }, {
                'type': 'LetterResize',
                'scale': (320, 320),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (960, 960)
            }, {
                'type': 'LetterResize',
                'scale': (960, 960),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }],
                    [{
                        'type': 'mmdet.RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'mmdet.RandomFlip',
                        'prob': 0.0
                    }], [{
                        'type': 'mmdet.LoadAnnotations',
                        'with_bbox': True
                    }],
                    [{
                        'type':
                        'mmdet.PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'pad_param', 'flip', 'flip_direction')
                    }]])
]
data_root = '/home/neau/sdb/datasets/leafs/'
train_ann_file = 'annotations/instances_train2017.json'
train_data_prefix = 'train2017/'
val_ann_file = 'annotations/instances_val2017.json'
val_data_prefix = 'val2017/'
num_classes = 2
train_batch_size_per_gpu = 4
train_num_workers = 4
persistent_workers = True
anchors = [[(12, 16), (19, 36), (40, 28)], [(36, 75), (76, 55), (72, 146)],
           [(142, 110), (192, 243), (459, 401)]]
base_lr = 0.01
max_epochs = 300
num_epoch_stage2 = 30
val_interval_stage2 = 1
model_test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.65),
    max_per_img=300)
img_scale = (640, 640)
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 1
val_num_workers = 2
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=1,
    img_size=640,
    size_divisor=32,
    extra_pad_ratio=0.5)
strides = [8, 16, 32]
num_det_layers = 3
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
max_translate_ratio = 0.2
scaling_ratio_range = (0.1, 2.0)
mixup_prob = 0.15
randchoice_mosaic_prob = [0.8, 0.2]
mixup_alpha = 8.0
mixup_beta = 8.0
loss_cls_weight = 0.3
loss_bbox_weight = 0.05
loss_obj_weight = 0.7
simota_candidate_topk = 10
simota_iou_weight = 3.0
simota_cls_weight = 1.0
prior_match_thr = 4.0
obj_level_weights = [4.0, 1.0, 0.4]
lr_factor = 0.1
weight_decay = 0.0005
save_epoch_intervals = 1
max_keep_ckpts = 3
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv7Backbone',
        arch='X',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.4,
            block_ratio=0.4,
            num_blocks=3,
            num_convs_in_block=2),
        upsample_feats_cat_first=False,
        in_channels=[640, 1280, 1280],
        out_channels=[160, 320, 640],
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        use_repconv_outs=False),
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=2,
            in_channels=[320, 640, 1280],
            featmap_strides=[8, 16, 32],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[[[31, 29], [36, 44], [49, 36]],
                        [[50, 48], [45, 65], [71, 42]],
                        [[60, 57], [61, 78], [80, 62]]],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.0075000000000000015),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=0.05,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.7),
        prior_match_thr=4.0,
        obj_level_weights=[4.0, 1.0, 0.4],
        simota_candidate_topk=10,
        simota_iou_weight=3.0,
        simota_cls_weight=1.0),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True)
]
mosiac4_pipeline = [
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.2,
        scaling_ratio_range=(0.1, 2.0),
        border=(-320, -320),
        border_val=(114, 114, 114))
]
mosiac9_pipeline = [
    dict(
        type='Mosaic9',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.2,
        scaling_ratio_range=(0.1, 2.0),
        border=(-320, -320),
        border_val=(114, 114, 114))
]
randchoice_mosaic_pipeline = dict(
    type='RandomChoice',
    transforms=[[{
        'type':
        'Mosaic',
        'img_scale': (640, 640),
        'pad_val':
        114.0,
        'pre_transform': [{
            'type': 'LoadImageFromFile',
            'file_client_args': {
                'backend': 'disk'
            }
        }, {
            'type': 'LoadAnnotations',
            'with_bbox': True
        }]
    }, {
        'type': 'YOLOv5RandomAffine',
        'max_rotate_degree': 0.0,
        'max_shear_degree': 0.0,
        'max_translate_ratio': 0.2,
        'scaling_ratio_range': (0.1, 2.0),
        'border': (-320, -320),
        'border_val': (114, 114, 114)
    }],
                [{
                    'type':
                    'Mosaic9',
                    'img_scale': (640, 640),
                    'pad_val':
                    114.0,
                    'pre_transform': [{
                        'type': 'LoadImageFromFile',
                        'file_client_args': {
                            'backend': 'disk'
                        }
                    }, {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }]
                }, {
                    'type': 'YOLOv5RandomAffine',
                    'max_rotate_degree': 0.0,
                    'max_shear_degree': 0.0,
                    'max_translate_ratio': 0.2,
                    'scaling_ratio_range': (0.1, 2.0),
                    'border': (-320, -320),
                    'border_val': (114, 114, 114)
                }]],
    prob=[0.8, 0.2])
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type':
            'Mosaic',
            'img_scale': (640, 640),
            'pad_val':
            114.0,
            'pre_transform': [{
                'type': 'LoadImageFromFile',
                'file_client_args': {
                    'backend': 'disk'
                }
            }, {
                'type': 'LoadAnnotations',
                'with_bbox': True
            }]
        }, {
            'type': 'YOLOv5RandomAffine',
            'max_rotate_degree': 0.0,
            'max_shear_degree': 0.0,
            'max_translate_ratio': 0.2,
            'scaling_ratio_range': (0.1, 2.0),
            'border': (-320, -320),
            'border_val': (114, 114, 114)
        }],
                    [{
                        'type':
                        'Mosaic9',
                        'img_scale': (640, 640),
                        'pad_val':
                        114.0,
                        'pre_transform': [{
                            'type': 'LoadImageFromFile',
                            'file_client_args': {
                                'backend': 'disk'
                            }
                        }, {
                            'type': 'LoadAnnotations',
                            'with_bbox': True
                        }]
                    }, {
                        'type': 'YOLOv5RandomAffine',
                        'max_rotate_degree': 0.0,
                        'max_shear_degree': 0.0,
                        'max_translate_ratio': 0.2,
                        'scaling_ratio_range': (0.1, 2.0),
                        'border': (-320, -320),
                        'border_val': (114, 114, 114)
                    }]],
        prob=[0.8, 0.2]),
    dict(
        type='YOLOv5MixUp',
        alpha=8.0,
        beta=8.0,
        prob=0.15,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type':
                    'Mosaic',
                    'img_scale': (640, 640),
                    'pad_val':
                    114.0,
                    'pre_transform': [{
                        'type': 'LoadImageFromFile',
                        'file_client_args': {
                            'backend': 'disk'
                        }
                    }, {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }]
                }, {
                    'type': 'YOLOv5RandomAffine',
                    'max_rotate_degree': 0.0,
                    'max_shear_degree': 0.0,
                    'max_translate_ratio': 0.2,
                    'scaling_ratio_range': (0.1, 2.0),
                    'border': (-320, -320),
                    'border_val': (114, 114, 114)
                }],
                            [{
                                'type':
                                'Mosaic9',
                                'img_scale': (640, 640),
                                'pad_val':
                                114.0,
                                'pre_transform': [{
                                    'type': 'LoadImageFromFile',
                                    'file_client_args': {
                                        'backend': 'disk'
                                    }
                                }, {
                                    'type': 'LoadAnnotations',
                                    'with_bbox': True
                                }]
                            }, {
                                'type': 'YOLOv5RandomAffine',
                                'max_rotate_degree': 0.0,
                                'max_shear_degree': 0.0,
                                'max_translate_ratio': 0.2,
                                'scaling_ratio_range': (0.1, 2.0),
                                'border': (-320, -320),
                                'border_val': (114, 114, 114)
                            }]],
                prob=[0.8, 0.2])
        ]),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/home/neau/sdb/datasets/leafs/',
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type':
                    'Mosaic',
                    'img_scale': (640, 640),
                    'pad_val':
                    114.0,
                    'pre_transform': [{
                        'type': 'LoadImageFromFile',
                        'file_client_args': {
                            'backend': 'disk'
                        }
                    }, {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }]
                }, {
                    'type': 'YOLOv5RandomAffine',
                    'max_rotate_degree': 0.0,
                    'max_shear_degree': 0.0,
                    'max_translate_ratio': 0.2,
                    'scaling_ratio_range': (0.1, 2.0),
                    'border': (-320, -320),
                    'border_val': (114, 114, 114)
                }],
                            [{
                                'type':
                                'Mosaic9',
                                'img_scale': (640, 640),
                                'pad_val':
                                114.0,
                                'pre_transform': [{
                                    'type': 'LoadImageFromFile',
                                    'file_client_args': {
                                        'backend': 'disk'
                                    }
                                }, {
                                    'type': 'LoadAnnotations',
                                    'with_bbox': True
                                }]
                            }, {
                                'type': 'YOLOv5RandomAffine',
                                'max_rotate_degree': 0.0,
                                'max_shear_degree': 0.0,
                                'max_translate_ratio': 0.2,
                                'scaling_ratio_range': (0.1, 2.0),
                                'border': (-320, -320),
                                'border_val': (114, 114, 114)
                            }]],
                prob=[0.8, 0.2]),
            dict(
                type='YOLOv5MixUp',
                alpha=8.0,
                beta=8.0,
                prob=0.15,
                pre_transform=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk')),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='RandomChoice',
                        transforms=[[{
                            'type':
                            'Mosaic',
                            'img_scale': (640, 640),
                            'pad_val':
                            114.0,
                            'pre_transform': [{
                                'type': 'LoadImageFromFile',
                                'file_client_args': {
                                    'backend': 'disk'
                                }
                            }, {
                                'type': 'LoadAnnotations',
                                'with_bbox': True
                            }]
                        }, {
                            'type': 'YOLOv5RandomAffine',
                            'max_rotate_degree': 0.0,
                            'max_shear_degree': 0.0,
                            'max_translate_ratio': 0.2,
                            'scaling_ratio_range': (0.1, 2.0),
                            'border': (-320, -320),
                            'border_val': (114, 114, 114)
                        }],
                                    [{
                                        'type':
                                        'Mosaic9',
                                        'img_scale': (640, 640),
                                        'pad_val':
                                        114.0,
                                        'pre_transform': [{
                                            'type': 'LoadImageFromFile',
                                            'file_client_args': {
                                                'backend': 'disk'
                                            }
                                        }, {
                                            'type': 'LoadAnnotations',
                                            'with_bbox': True
                                        }]
                                    }, {
                                        'type': 'YOLOv5RandomAffine',
                                        'max_rotate_degree': 0.0,
                                        'max_shear_degree': 0.0,
                                        'max_translate_ratio': 0.2,
                                        'scaling_ratio_range': (0.1, 2.0),
                                        'border': (-320, -320),
                                        'border_val': (114, 114, 114)
                                    }]],
                        prob=[0.8, 0.2])
                ]),
            dict(type='YOLOv5HSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ],
        metainfo=dict(
            classes=('round', 'shape'), palette=[(20, 220, 60),
                                                 (20, 190, 60)])))
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/home/neau/sdb/datasets/leafs/',
        test_mode=True,
        data_prefix=dict(img='val/images/'),
        ann_file='val/annotation_coco.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5),
        metainfo=dict(
            classes=('round', 'shape'), palette=[(20, 220, 60),
                                                 (20, 190, 60)])))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/home/neau/sdb/datasets/leafs/',
        test_mode=True,
        data_prefix=dict(img='val/images/'),
        ann_file='val/annotation_coco.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5),
        metainfo=dict(
            classes=('round', 'shape'), palette=[(20, 220, 60),
                                                 (20, 190, 60)])))
param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=4),
    constructor='YOLOv7OptimWrapperConstructor')
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='/home/neau/sdb/datasets/leafs/val/annotation_coco.json',
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='/home/neau/sdb/datasets/leafs/val/annotation_coco.json',
    metric='bbox')
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=1,
    dynamic_intervals=[(270, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
class_name = ('round', 'shape')
metainfo = dict(
    classes=('round', 'shape'), palette=[(20, 220, 60), (20, 190, 60)])