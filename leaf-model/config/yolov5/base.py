
max_epochs = 300

train_num_workers = 1



model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=2),
    )
)



train_dataloader = dict(
    num_workers=train_num_workers,
    )



default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=1, save_best='auto',save_param_scheduler=False),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=50)
)

train_cfg = dict(max_epochs=max_epochs, val_interval=10)



visualizer = dict(
    vis_backends= [
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',         
            init_kwargs={
                'project' : 'soybean-leaf',
        })]
    )




