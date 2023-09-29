# 分析数据集


# 验证和训练
python /home/neau/sdb/mmyolo/tools/analysis_tools/dataset_analysis.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-s/yolov8-s.py 
python tools/analysis_tools/dataset_analysis.py /home/neau/sdb/ag-pro/leaf_model/config/yolov5/yolov5-l-p6.py --val-dataset


# 数据强化步骤

python /home/neau/sdb/mmyolo/tools/analysis_tools/browse_dataset.py \
    /home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-ag-mosaic9-mixup.py \
    -m pipeline 
    

# 打印配置文件

python /home/neau/sdb/mmyolo/tools/misc/print_config.py  /home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l-p6.py    --save-path ./test_config.py --cfg-options model.test_cfg.nms.iou_threshold=$nms_iou_threshold work_dirs=/home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l-p6-nms$nms_iou_threshold




# 测试结果
CUDA_VISIBLE_DEVICES=-1 
python /home/neau/sdb/mmdetection/tools/analysis_tools/analyze_results.py /home/neau/sdb/mmdetection/tools/test.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/out.pkl ./aaa --topk 10

PYTHONPATH=/home/neau/sdb/ag-pro/leaf_model python /home/neau/sdb/mmdetection/tools/analysis_tools/analyze_results.py  /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/out1.pkl aaa  --topk 10


# 生成测试结果pku
PYTHONPATH=/home/neau/sdb/ag-pro/leaf_model python /home/neau/sdb/mmdetection/tools/test.py        /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py        /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/best_coco_bbox_mAP_epoch_260.pth --out ./out1.pkl

tools/analysis_tools/analyze_results.py


