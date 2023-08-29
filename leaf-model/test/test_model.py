
from mmengine.config import Config
from mmengine.registry import Registry

from mmyolo.registry import MODELS

import torch

config_file = '/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5_l.py'
cfg = Config.fromfile(config_file)

img = torch.rand(1,3,640,640)

model = MODELS.build(cfg.model.backbone)

print(model)



