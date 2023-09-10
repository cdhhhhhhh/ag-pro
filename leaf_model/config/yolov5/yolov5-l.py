_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco.py',
]

add_config = '/home/neau/sdb/ag-pro/leaf_model/config/yolov5/base.py'
project_name = 'yolov5-l'



train_batch_size_per_gpu = 4

model = dict(
        bbox_head=dict(
    loss_cls=dict(loss_weight= 0.3 *
        (2 / 80 * 3 / 3)),  
        )
)
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
)

