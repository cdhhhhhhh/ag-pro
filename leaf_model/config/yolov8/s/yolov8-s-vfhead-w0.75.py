_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s.py',
]



model = dict(
     bbox_head=dict(
     use_vfl = True,
    loss_cls_vfl=dict(
            type='mmdet.VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            reduction='none',
            iou_weighted=True,
            loss_weight=1),
))


