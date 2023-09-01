import torch

from mmengine.registry import HOOKS
from mmengine.hooks import Hook

import pandas as pd

total = ''

@HOOKS.register_module()
class MySelfExpHook(Hook):
   

    def __init__(self, interval=50):
        self.interval = interval


                
    def after_val_epoch(self, runner, metrics):
        # GPU使用率
        
        print('test')
        # 保存当前训练实时指标
        pass
    
    
    
    def after_run(self, runner):
        pass
        
        # 训练数据
        runner.message_hub.state_dict()['log_scalars']['train/loss'].data

        # 验证数据
        
        
        # 最优数据
        columns = ['name','p','r','f1','map0.5','map0.75','map0.5-0.95','gpu','leaf_map0.5','round_map0.5','epoch']
        name = runner.cfg.project_name
        
        # pd.DataFrame([],columns=[])
        
        # 可视化结果记录
        
        # 混淆矩阵
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    