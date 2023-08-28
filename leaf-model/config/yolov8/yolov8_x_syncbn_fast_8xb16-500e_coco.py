_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco.py',
]




train_batch_size_per_gpu = 2
train_num_workers = 2

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
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
)


visualizer = dict(
    vis_backends= _base_.vis_backends + [
        dict(
            type='WandbVisBackend',         
            init_kwargs={
                'project' : 'soybean-leaf',
                'name' : 'yolov8-x'
        })]
    )



default_hooks = dict(
    checkpoint=dict(save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50)
)
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
