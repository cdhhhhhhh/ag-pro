
train_num_workers = 1

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
    num_workers=train_num_workers,
)




default_hooks = dict(
    checkpoint=dict(save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50)
)


visualizer = dict(
    vis_backends= [
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',         
            init_kwargs={
                'project' : 'soybean-leaf',
        })]
    )




