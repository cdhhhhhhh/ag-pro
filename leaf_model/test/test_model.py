
from mmengine.config import Config
from mmengine.registry import Registry

from mmyolo.registry import MODELS


import torch

from projects.EfficientDet.efficientdet.bifpn import BiFPN
from mmyolo.models.backbones.csp_darknet import YOLOv8CSPDarknet

# config_file = '/home/neau/sdb/ag-pro/leaf_model/config/yolov5/yolov5_l.py'
# cfg = Config.fromfile(config_file)

img = torch.rand(1,3,640,640)

model = YOLOv8CSPDarknet(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=1.0,
        last_stage_out_channels=512,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        widen_factor=1.0
        )


output = model(img)
print('backbone' , [i.shape for i in output])

bifpn = BiFPN(
    num_stages=6,
    in_channels=[256,
            512,
            512,],
    out_channels=160,
    start_level=0,
)




output = bifpn(output)


print('neck' , [i.shape for i in output])



