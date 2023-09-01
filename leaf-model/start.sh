# 分析数据集


# 验证和训练
python tools/analysis_tools/dataset_analysis.py /home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-l-p6.py 
python tools/analysis_tools/dataset_analysis.py /home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-l-p6.py --val-dataset


# 数据强化步骤

python tools/analysis_tools/browse_dataset.py \
    /home/neau/sdb/mmyolo/task/leafs/wheet/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py \
    -m pipeline \
    

# 打印配置文件

python /home/neau/sdb/mmyolo/tools/misc/print_config.py /home/neau/sdb/ag-pro/wheat/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py  --save-path output.json














