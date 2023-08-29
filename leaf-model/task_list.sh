
task_array=(
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-l-p6.py" 
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-l.py"
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-x.py"
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov5/yolov5-x-p6.py" 
    # "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-x.py"
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-l.py"
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-s.py"
    "/home/neau/sdb/ag-pro/leaf-model/config/yolov8/yolov8-m.py"
    )


for task in "${task_array[@]}"; do
  echo "start!!!!--------$task"
  tsp "./dist_train.sh" $task 4 --amp 
  sleep 1
done



