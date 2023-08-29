# 分析数据集
python tools/analysis_tools/dataset_analysis.py /home/neau/sdb/mmyolo/task/leafs/wheet/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py  





python tools/analysis_tools/browse_dataset.py \
    /home/neau/sdb/mmyolo/task/leafs/wheet/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py \
    -m pipeline \
    



python /home/neau/sdb/mmyolo/tools/misc/print_config.py /home/neau/sdb/ag-pro/wheat/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py  --save-path output.json


tsp /home/neau/sdb/mmyolo/tools/dist_train.sh /home/neau/sdb/ag-pro/wheat/yolov5_l-v61_syncbn_fast_8xb16-300e_coco.py 4 --amp

python /home/neau/sdb/mmyolo/tools/misc/print_config.py 







