_base_ = [
    '/home/neau/sdb/mmyolo/configs/yolov5/yolov5_x-p6-v62_syncbn_fast_8xb16-300e_coco.py',
]


add_config = '/home/neau/sdb/ag-pro/leaf-model/config/yolov5/base.py'

project_name = 'yolov5-x-p6'


train_batch_size_per_gpu = 2

model = dict(
    loss_cls=dict(loss_weight= 0.3 *
        (2 / 80 * 3 / 4)),  
)





train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
)


_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu




# load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco/yolov5_l-v61_syncbn_fast_8xb16-300e_coco_20220917_031007-096ef0eb.pth'
