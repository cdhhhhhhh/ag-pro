_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-ag-mixup.py',
]


model = dict(
    bbox_head= dict(
        head_module = dict(
            num_classes = 1
        )
    ),
    train_cfg = dict(
        assigner = dict(
            num_classes = 1,
        )
    )
)



data_root = '/home/neau/trainset/crop_leafs_dataset-single'
metainfo = dict(classes=('leaf'))

_base_.train_dataloader.dataset.metainfo = metainfo
_base_.train_dataloader.dataset.data_root = data_root
_base_.train_dataloader.dataset.ann_file = 'train_annotations.json'
_base_.train_dataloader.dataset.data_prefix = dict(img='./')

_base_.val_dataloader.dataset.metainfo = metainfo
_base_.val_dataloader.dataset.data_root = data_root
_base_.val_dataloader.dataset.ann_file = 'val_annotations.json'
_base_.val_dataloader.dataset.data_prefix = dict(img='./')

_base_.test_dataloader = _base_.val_dataloader

_base_.val_evaluator.ann_file = data_root + '/val_annotations.json'
_base_.val_evaluator.proposal_nums = (100, 300, 1000) ## coco设定

_base_.test_evaluator = _base_.val_evaluator


