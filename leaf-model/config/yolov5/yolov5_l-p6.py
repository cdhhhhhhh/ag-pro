_base_= [
    '/home/neau/sdb/mmyolo/configs/yolov5/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py',
]

loss_cls_weight = 0.3
loss_obj_weight = 0.7
num_det_layers = _base_.num_det_layers

num_classes = 2



max_epochs = 300
train_batch_size_per_gpu = 2
train_num_workers = 1



model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=2),
        loss_cls=dict(loss_weight=loss_cls_weight *
                (num_classes / 80 * 3 / num_det_layers)),  
    )
)




train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
)


_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu


default_hooks = dict(
    checkpoint=dict(interval=10, 
                    max_keep_ckpts=1, 
                    save_best='auto', 
                    save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50)
)


train_cfg = dict(max_epochs=max_epochs, val_interval=5)




visualizer = dict(
    vis_backends= _base_.vis_backends + [
        dict(
            type='WandbVisBackend',         
            init_kwargs={
                'project' : 'soybean-leaf',
                'name' : 'yolov5-l-p6'
        })]
)
