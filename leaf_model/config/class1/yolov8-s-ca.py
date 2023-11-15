_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/class1/yolov8-s-ag-no.py',
]


model = dict(
    backbone=dict(
        channel_attention=True,
        use_conv='dcn',
        use_msppf=True,
    ),
    neck=dict(
        _delete_=True,
        in_channels=[
            256,
            512,
            1024,
        ],
        out_channels=[
            256,
            512,
            1024,
        ],
        type='YOLOv8AFPN',
        ca=True,
        widen_factor=0.5),
)


train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    dynamic_intervals=None,
    val_interval=1
)


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))



default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=-1
    ),
    param_scheduler=None
)
