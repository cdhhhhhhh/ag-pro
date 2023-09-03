import torch

from mmengine.registry import HOOKS
from mmengine.hooks import Hook

import pandas as pd
import subprocess



def get_gpu_memory():
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024  # MB
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values
    except Exception as e:
        print(f"Error: {e}")
        return None


@HOOKS.register_module()
class MySelfExpHook(Hook):
   

    def __init__(self):
        pass


                
    def after_val_epoch(self, runner, metrics):
        # GPU使用率
        gpus_memory = get_gpu_memory()
        runner.message_hub.update_scalar('gpu', max([round(i / 11264 , 4) for i in gpus_memory]), 1)
    
    
    def after_run(self, runner):
        log_scalars = runner.message_hub.state_dict()['log_scalars']
        val_interval = runner.cfg.train_cfg.val_interval
        max_epochs = runner.cfg.train_cfg.max_epochs
        total_pathname = '/home/neau/sdb/ag-pro/leaf-model/totals.csv'

        # 训练数据

        train_keys = list(filter(lambda item: 'train' in item,log_scalars.keys()))
        pd.DataFrame({key: log_scalars[key].data[0] for key in train_keys }).to_csv(runner.cfg.work_dir + '/train_log.csv')


        # 验证数据
        val_keys = list(filter(lambda item: 'val/coco' in item,log_scalars.keys()))
        val_dic = {key: log_scalars[key].data[0] for key in val_keys }
        val_dic['epoch'] = [ val_interval*i for i in range(1, len(log_scalars[val_keys[0]].data[0]) + 1)]
        val_pd = pd.DataFrame(val_dic)
        val_pd.to_csv(runner.cfg.work_dir + '/val_log.csv')

        
        # 最优数据
        best_dic = val_pd[val_pd['val/coco/bbox_mAP_50'] == val_pd['val/coco/bbox_mAP_50'].max()].to_dict()
        best_i = val_pd[val_pd['val/coco/bbox_mAP_50'] == val_pd['val/coco/bbox_mAP_50'].max()].index.values[0]
        best_dic['gpu'] = { best_i : log_scalars['gpu'].data[0].max() }
        best_dic['name'] = { best_i : runner.cfg.project_name }
        
        total_pd = pd.read_csv(total_pathname)
        
        
        if len(total_pd[total_pd['name'] == runner.cfg.project_name]) == 0:
            total_pd = pd.concat([ pd.read_csv(total_pathname), pd.DataFrame(best_dic) ],axis=0, ignore_index=True)
        else:
            total_pd[total_pd['name'] == runner.cfg.project_name] = pd.DataFrame(best_dic)[total_pd.columns]
        
        total_pd.to_csv(total_pathname, index=False)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    