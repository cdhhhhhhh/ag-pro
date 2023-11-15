# 分析数据集


# 验证和训练
python /home/neau/sdb/mmyolo/tools/analysis_tools/dataset_analysis.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-s/yolov8-s.py 
python tools/analysis_tools/dataset_analysis.py /home/neau/sdb/ag-pro/leaf_model/config/yolov5/yolov5-l-p6.py --val-dataset


# 数据强化步骤

python /home/neau/sdb/mmyolo/tools/analysis_tools/browse_dataset.py \
    /home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-p5-4.py \
    -m pipeline 


python /home/neau/sdb/mmdet/tools/analysis_tools/browse_dataset.py \
    /home/neau/sdb/mmdetection/configs/soft_teacher/soft-teacher_yolov8s.py
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




python /home/neau/sdb/mmyolo/demo/featmap_vis_demo.py /home/neau/trainset/images/d0f0ba7a-20220701-crop-0-4-0.jpg /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-s/yolov8-s.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-s/best_coco_bbox_mAP_epoch_498.pth --target-layers backbone --channel-reduction select_max --device cpu


python /home/neau/sdb/mmyolo/demo/featmap_vis_demo.py /home/neau/trainset/images/d0f0ba7a-20220701-crop-0-4-0.jpg /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-s-ag-mixup/yolov8-s-ag-mixup.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-s-ag-mixup/best_coco_bbox_mAP_epoch_500.pth --target-layers backbone neck --channel-reduction squeeze_mean --device cpu

# 正负样本

python /home/neau/sdb/mmyolo/projects/assigner_visualization/assigner_visualization.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-s-backbone/yolov8-s-backbone-test.py -c /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-s-backbone/best_coco_round_precision_epoch_230.pth 


# 预测结果
python /home/neau/sdb/mmyolo/tools/test.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --show --show-dir ./ouput



python /home/neau/sdb/mmyolo/tools/analysis_tools/optimize_anchors.py /home/neau/sdb/ag-pro/leaf_model/config/head/yolov8-s-head-v5-4.py



python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0613 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --to-labelme


python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0625 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --to-labelme --out-dir ./output/0625
python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0620 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --to-labelme --out-dir ./output/0620


python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0701 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --to-labelme --out-dir ./output/0701 --score-thr 0.5
python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0708 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --to-labelme --out-dir ./output/0708 --score-thr 0.5
python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0713 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --to-labelme --out-dir ./output/0713 --score-thr 0.5

python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0613 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --to-labelme --out-dir ./output/0613 --score-thr 0.5
python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0620 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --to-labelme --out-dir ./output/0620 --score-thr 0.5
python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0625 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth --to-labelme --out-dir ./output/0625 --score-thr 0.5

python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0701 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth  --out-dir ./output/0701 --score-thr 0.5
python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0708 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth  --out-dir ./output/0708 --score-thr 0.5
python /home/neau/sdb/mmyolo/demo/large_image_demo.py /home/neau/trainset/pix4d-0713 /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth  --out-dir ./output/0713 --score-thr 0.5


python /home/neau/sdb/mmyolo/demo/image_demo.py /home/neau/trainset/leaf-o /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/yolov8-l.py /home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l/epoch_300.pth  --out-dir ./output/leaf-o



