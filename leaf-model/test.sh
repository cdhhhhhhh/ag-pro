nms_iou_threshold_arr=($(seq 0.4 0.05 0.9))
nms_method_arr=("nms" "soft-nms")

cfg_options_name=("nms_method_arr")

# 使用间接引用来获取数组
inner_array_str=${cfg_options_name[0]}
indirect_reference_array="${!inner_array_str[@]}"

# 输出间接引用数组的内容
echo ${indirect_reference_array[@]}  # 输出 "nms soft-nms"
