nms_iou_threshold_arr=($(seq 0.4 0.05 0.9))

nms_method_arr=("nms"
                "soft-nms")

task_arr=""

for nms_method in ${nms_method_arr[@]}; do
  for nms_iou_threshold in ${nms_iou_threshold_arr[@]}; do

    task_arr=${task_arr} model.test_cfg.nms.iou_threshold=${nms_iou_threshold}
    # echo model.test_cfg.nms.iou_threshold=$nms_iou_threshold


  done
done



echo ${task_arr}