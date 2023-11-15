_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]

widen_factor = 0.5
num_classes = 2
strides = [8, 16, 32]
model = dict(
    backbone=dict(
        channel_attention=True,
        use_conv='dcn',
        use_msppf=True,
    ),
    neck=dict(
        attention=dict(
            pos='last',
            type='ca'
        )
    ),
    bbox_head=dict(
        type='PPYOLOEHead',
        head_module=dict(
            type='PPYOLOEHeadModule',
            num_classes=num_classes,
            in_channels=[256,
                         512,
                         1024,],
            widen_factor=widen_factor,
            featmap_strides=strides,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
            act_cfg=dict(type='SiLU', inplace=True),
            num_base_priors=1),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='giou',
            bbox_format='xyxy',
            reduction='mean',
            loss_weight=2.5,
            return_iou=False),
        # Since the dflloss is implemented differently in the official
        # and mmdet, we're going to divide loss_weight by 4.
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=0.5 / 4)),
    train_cfg=dict(
        initial_epoch=60,
        initial_assigner=dict(
            type='BatchATSSAssigner',
            num_classes=num_classes,
            topk=9,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            topk=13,
            alpha=1,
            beta=6,
            eps=1e-9)),
)
