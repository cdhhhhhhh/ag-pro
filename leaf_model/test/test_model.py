
from mmengine.config import Config
from mmengine.registry import Registry

from mmyolo.registry import MODELS


import torch

from projects.EfficientDet.efficientdet.bifpn import BiFPN
from mmyolo.models.backbones.csp_darknet import YOLOv8CSPDarknet

from leaf_model_tools.AFPN import YOLOv8AFPN
from mmpretrain.models.backbones.densenet import DenseNet
from leaf_model_tools.GDNeck import GDSNeck

# config_file = '/home/neau/sdb/ag-pro/leaf_model/config/yolov5/yolov5_l.py'
# cfg = Config.fromfile(config_file)

img = torch.rand(1, 3, 640, 640)


model = YOLOv8CSPDarknet(
    act_cfg=dict(inplace=True, type='SiLU'),
    arch='P5',
    deepen_factor=0.33,
    out_indices = (1,2, 3, 4),
    last_stage_out_channels=1024,
    norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
    widen_factor=0.5,
)

# model = DenseNet(arch='201',out_indices=(0, 1, 2, 3))


# model = ResNet(
#         depth=50,
#         in_channels=3,
#         out_indices=[4],  # 0: conv-1, x: stage-x
#         norm_cfg=dict(type='BN')
#         )
output = model(img)
print('backbone', [i.shape for i in output])


# bifpn = BiFPN(
#     num_stages=6,
#     in_channels=[256,
#             512,
#             512,],
#     out_channels=160,
#     start_level=0,
# )

# neck = YOLOv8AFPN(
#     in_channels=[256,
#                  512,
#                  1024,],
#     out_channels=[256,
#                  512,
#                  1024,],
#     widen_factor=0.5
# )

neck = GDSNeck(
    num_repeats=[12, 12, 12, 12,12, 12, 12, 12,12, 12, 12, 12,12, 12, 12, 12],
    channels_list=[0, 0, 0, 128, 256, 512, 1024,512,512,512,1024,512,512,512,1024,512,512,512],
    extra_cfg=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        depths=2,
        fusion_in=960,
        ppa_in=704,
        fusion_act=dict(type='ReLU6'),
        fuse_block_num=3,
        embed_dim_p=128,
        embed_dim_n=704,
        key_dim=8,
        num_heads=4,
        mlp_ratios=1,
        attn_ratios=2,
        c2t_stride=2,
        drop_path_rate=0.1,
        trans_channels=[128, 64, 128, 256],
        pool_mode='torch'
    ),
)
output = neck(output)
print('neck', [i.shape for i in output])

# output = bifpn(output)
