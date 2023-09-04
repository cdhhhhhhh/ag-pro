
task_array=(
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-l-p6.py" 
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-l.py"
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-x.py"
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-x-p6.py" 
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-x.py"
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-l.py"
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-s.py"
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-m.py"
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-x-p6.py"
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-l-p6.py"
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-s-p6.py"
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-m-p6.py"
    )

nms_iou_threshold_arr=($(seq 0.4 0.05 0.9))
nms_method_arr=("nms"
                "soft-nms")


for task in "${task_array[@]}"; do
  for nms_iou_threshold in "${nms_iou_threshold_arr[@]}"; do
    tsp "./dist_train.sh" $task 4 --amp --cfg-options model.test_cfg.nms.iou_threshold=$nms_iou_threshold_arr work_dir=/home/neau/sdb/ag-pro/leaf-model/work_dirs/yolov8-l-p6-nms$nms_iou_threshold_arr project_name=yolov8-l-p6-nms$nms_iou_threshold_arr
    
    echo "start!!!!--------$task"
    sleep 1
  done
done


# "./dist_train.sh" /home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-l-p6.py 4 --amp --cfg-options model.test_cfg.nms.iou_threshold=$nms_iou_threshold_arr work_dir=/home/neau/sdb/ag-pro/leaf-model/work_dirs/yolov8-l-p6-nms$nms_iou_threshold_arr project_name=yolov8-l-p6-nms$nms_iou_threshold_arr


# python /home/neau/sdb/mmyolo/tools/misc/print_config.py  /home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-l-p6.py    --save-path ./test_config.py --cfg-options model.test_cfg.nms.iou_threshold=$nms_iou_threshold_arr work_dirs=/home/neau/sdb/ag-pro/leaf-model/work_dirs/yolov8-l-p6-nms$nms_iou_threshold_arr
