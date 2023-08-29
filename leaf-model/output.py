default_hooks = dict(
    checkpoint=dict(save_param_scheduler=False),
    logger=dict(interval=50, type='LoggerHook'))
model = dict(
    bbox_head=dict(head_module=dict(num_classes=2)),
    train_cfg=dict(assigner=dict(num_classes=2)))
train_batch_size_per_gpu = 4
train_dataloader = dict(batch_size=4, num_workers=2)
train_num_workers = 2
