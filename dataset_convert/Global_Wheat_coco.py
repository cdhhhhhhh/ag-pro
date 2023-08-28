import pandas as pd

import mmcv
from mmengine.fileio import dump
from mmengine.utils import track_iter_progress

import os.path as osp




def convert_coco(images_dir,label_csv,label_output):
    
    images = []
    annotations = []
    pd_train = pd.read_csv(label_csv)
    ann_count = 0
    for i in track_iter_progress(range(pd_train.shape[0])):
        item = pd_train.iloc[i]
        image_name = item['image_name']
        BoxesString = item['BoxesString']

        if BoxesString == 'no_box':
            continue
        
        BoxesString_list = BoxesString.split(';')
        BoxesString_list = [[int(j) for j in i.split(' ')]
                            for i in BoxesString_list]

        img = mmcv.imread(osp.join(images_dir, image_name))
        height, width, _ = img.shape
        images.append(
            dict(id=i, file_name=image_name, height=height, width=width))

        for idx,boxx in enumerate(BoxesString_list):
            
            x_min, y_min, x_max, y_max = boxx
            w_bbox = x_max - x_min
            h_bbox = y_max - y_min
            
            annotations.append(dict(
                image_id=i,
                id=ann_count,
                category_id=0,
                bbox=[x_min, y_min, w_bbox, h_bbox],
                area=w_bbox * h_bbox,
                segmentation=[],
                iscrowd=0
            ))
            ann_count += 1
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'wheat'
        }])
    
    
    dump(coco_format_json, label_output)
    
    
    
    
if __name__ == '__main__':

    convert_coco('/home/neau/sdb/datasets/OpenDataLab___Global_Wheat/raw/gwhd_2021/images/',
                 '/home/neau/sdb/datasets/OpenDataLab___Global_Wheat/raw/gwhd_2021/competition_train.csv',
                 '/home/neau/sdb/datasets/OpenDataLab___Global_Wheat/raw/gwhd_2021/competition_train_coco.json'
                 )
    
    
    
    convert_coco('/home/neau/sdb/datasets/OpenDataLab___Global_Wheat/raw/gwhd_2021/images/',
                 '/home/neau/sdb/datasets/OpenDataLab___Global_Wheat/raw/gwhd_2021/competition_val.csv',
                 '/home/neau/sdb/datasets/OpenDataLab___Global_Wheat/raw/gwhd_2021/competition_val_coco.json'
                 )