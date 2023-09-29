_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l.py',
]

# custom_imports = dict(
#     imports=['projects.EfficientDet.efficientdet'], allow_failed_imports=False)

norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)

project_name = 'yolov8-l-bifpn'


model = dict(
    neck=dict(
        _delete_=True,
        type='projects.EfficientDet.efficientdet.BiFPN',
        num_stages=3,
        in_channels=[
            256,
            512,
            512,
            ],
        out_channels=512,
        start_level=0,
        # norm_cfg=norm_cfg
        ),
    bbox_head = dict(
        head_module = dict(
            in_channels = [
                512,
                512,
                512
            ]
        )
    )
)





# model = dict(
#     neck=dict(
#         type='BiFPN',
#         num_stages=6,
#         in_channels=[
#             256,
#             512,
#             512,
#             ],
#         out_channels=348,
#         start_level=0,
#         norm_cfg=norm_cfg),
#     bbox_head = dict(
#         head_module = dict(
#             in_channels = [
#                 348,
#                 348,
#                 348
#             ]
#         )
#     )
# )


