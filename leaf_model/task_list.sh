yolov5_arr=(
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov5/yolov5-l-p6.py" 
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov5/yolov5-l.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov5/yolov5-x.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov5/yolov5-x-p6.py" 
)



yolov8_arr=(
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-x.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-m.py"
)
yolov8_cbam_arr=(
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-x-cbam-backbone.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l-cbam-backbone.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-cbam-backbone.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-m-cbam-backbone.py"
)

yolov8_p6_arr=(
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-x-p6.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l-p6.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-p6.py"
    "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-m-p6.py"
)

# yolov8_p6_1024_arr=(
#     "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-x-p6-1024.py"
#     "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-l-p6-1024.py"
#     "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-p6-1024.py"
#     "/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-m-p6-1024.py"
# )


# NMS 

# yolov8_p6_nms=${yolov8_p6_arr[1]}
# nms_iou_threshold_arr=($(seq 0.3 0.1 0.9))
# nms_method_arr=(
#                 # "nms"
#                 "soft_nms"
#                 # "nms_match"
#                 )



# for nms_method in ${nms_method_arr[@]}; do
#   for nms_iou_threshold in ${nms_iou_threshold_arr[@]}; do

#     tsp "./dist_train.sh" ${yolov8_p6_nms} 4 --amp --cfg-options model.test_cfg.nms.type=${nms_method} model.test_cfg.nms.iou_threshold=$nms_iou_threshold work_dir=/home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l-p6-${nms_method}$nms_iou_threshold project_name=yolov8-l-p6-${nms_method}$nms_iou_threshold
#     # echo --cfg-options model.test_cfg.nms.type=${nms_method} model.test_cfg.nms.iou_threshold=$nms_iou_threshold work_dir=/home/neau/sdb/ag-pro/leaf_model/work_dirs/yolov8-l-p6-${nms_method}$nms_iou_threshold project_name=yolov8-l-p6-${nms_method}$nms_iou_threshold
#   done
# done




# model
# task_array=(${yolov5_arr} ${yolov8_arr} ${yolov8_p6_arr} ${yolov8_p6_1024_arr})


for task in ${yolov8_cbam_arr[@]}; do
    tsp "./dist_train.sh" $task 4 --amp
done




# 注意力





# neck




# IOU




# TTA



# 多尺度


# 数据增强



