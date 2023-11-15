_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-mos.py',
]

num_classes = 2
widen_factor = 0.5
strides = [4,8]

anchors = [
    [(26, 37), (41, 28), (39, 47)],
]

loss_cls_weight = 0.5

num_det_layers = 1

loss_bbox_weight = 0.05
loss_obj_weight = 1.0
img_scale = (640, 640)

prior_match_thr = 4
obj_level_weights = [1.]

model = dict(
    neck=dict(
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        out_channels=[
            128,
            256,
        ],
    ),
    bbox_head=dict(
        _delete_=True,
        type='YOLOv5Head',
        head_module=dict(
             type='YOLOv5HeadModule',
            num_classes=num_classes,
            in_channels=[128,256],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight *
            (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight *
            ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights),
)
