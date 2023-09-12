_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l.py',
]


project_name = 'yolov8-l-repeat-dataset-5'

origin_dataset = _base_.train_dataloader.dataset

train_dataloader = dict(
    dataset = dict(
        type = 'RepeatDataset',
        dataset = origin_dataset,
        times = 5
    )
)