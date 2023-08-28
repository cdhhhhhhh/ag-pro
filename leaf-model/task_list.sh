
task_array=(
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5_l-p6.py" 
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5_l.py"
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5_x.py"
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5_x-p6.py" 
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py"
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco.py"
    )


for task in "${task_array[@]}"; do
  echo "start!!!!--------$task"
  tsp "./dist_train.sh" $task 4 --amp 
  sleep 1
done



