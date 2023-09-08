
from mmengine.config import Config
from mmengine.registry import Registry

from mmyolo.registry import MODELS

import torch

# config_file = '/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5_l.py'
# cfg = Config.fromfile(config_file)

# img = torch.rand(1,3,640,640)

# model = MODELS.build(cfg.model.backbone)

# print(model)



from mmyolo.models import YOLOv8CSPDarknetMySelf,YOLOv5CSPDarknet,YOLOv8PAFPN,YOLOv8Head
import torch



# model = YOLOv8CSPDarknetMySelf(act_cfg=dict(inplace=True, type='SiLU'),
#         arch='P6',
#         deepen_factor=1.0,
#         # last_stage_out_channels=1024,
#         norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
#         widen_factor=1.25,
#         out_indices = (2, 3, 4, 5),
#         )

# neck = YOLOv8PAFPN(act_cfg=dict(inplace=True, type='SiLU'),
#         deepen_factor=1.0,
#         in_channels=[
#             256,
#             512,
#             768,
#             1024,
#         ],
#         norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
#         num_csp_blocks=3,
#         out_channels=[
#             256,
#             512,
#             768,
#             1024,   
#         ],
#         widen_factor=1.25)

# head = YOLOv8Head( bbox_head=dict(
#         bbox_coder=dict(type='DistancePointBBoxCoder'),
#         head_module=dict(
#             act_cfg=dict(inplace=True, type='SiLU'),
#             featmap_strides=[
#                 8,
#                 16,
#                 32,
#             ],
#             in_channels=[
#                 256,
#                 512,
#                 512,
#             ],
#             norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
#             num_classes=2,
#             reg_max=16,
#             type='YOLOv8HeadModule',
#             widen_factor=1.25),
#         loss_bbox=dict(
#             bbox_format='xyxy',
#             iou_mode='ciou',
#             loss_weight=7.5,
#             reduction='sum',
#             return_iou=False,
#             type='IoULoss'),
#         loss_cls=dict(
#             loss_weight=0.5,
#             reduction='none',
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=True),
#         loss_dfl=dict(
#             loss_weight=0.375,
#             reduction='mean',
#             type='mmdet.DistributionFocalLoss'),
#         prior_generator=dict(
#             offset=0.5, strides=[
#                 8,
#                 16,
#                 32,
#             ], type='mmdet.MlvlPointGenerator'),
#         type='YOLOv8Head'))

# model.eval()
# inputs = torch.rand(1, 3, 1024, 1024)
# level_outputs = model(inputs)
# # level_outputs = neck(level_outputs)

# for level_out in level_outputs:
#     print(tuple(level_out.shape))
    
    


from mmyolo.models.necks.yolov5_pafpn import YOLOv5PAFPN

fpn = YOLOv5PAFPN(
         act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=1.0,
        in_channels=[
            256,
            512,
            768,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            768,
            1024,
        ],
        widen_factor=1.0)


print(fpn)


