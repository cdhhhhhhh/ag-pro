_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov5/yolov5_x-v61_syncbn_fast_8xb16-300e_coco.py',
]




max_epochs = 300
train_batch_size_per_gpu = 2
train_num_workers = 1



model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=2),
        # prior_generator=dict(base_sizes=anchors)  
    ))



train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    )


_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu





default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=1, save_best='auto',save_param_scheduler=False),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=50)
)

train_cfg = dict(max_epochs=max_epochs, val_interval=10)



visualizer = dict(
    vis_backends= _base_.vis_backends + [
        dict(
            type='WandbVisBackend',         
            init_kwargs={
                'project' : 'soybean-leaf',
                'name' : 'yolov5-x'
        })]
    )



