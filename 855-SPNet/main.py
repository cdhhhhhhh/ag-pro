


import os
import random
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar
from pyproj import Transformer

try:
    from sahi.slicing import slice_image
except ImportError:
    raise ImportError('Please run "pip install -U sahi" '
                      'to install sahi first for large image inference.')


from mmyolo.utils import switch_to_deploy

from mmyolo.utils.large_image import merge_results_by_nms, shift_predictions
from mmyolo.utils.misc import get_file_list


import pandas as pd
import cv2
import numpy as np
import tqdm
import glob
from lib import calc_degree, calc_scalc, convert_latlon_to_utm, get_distance_point_to_line, time_to_utc, get_image_metadata
from datetime import datetime, timezone
import json


def parse_args():
    parser = ArgumentParser(
        description='SPNet模型大图推理')
    parser.add_argument(
        'img', help='推理图像路径')
    parser.add_argument('config', help='模型文件')
    parser.add_argument('checkpoint', help='模型权重')
    parser.add_argument(
        '--project', default='project_default', help='项目名')
    parser.add_argument(
        '--out-dir', default='./output', help='输出路径')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--patch-size', type=int, default=640, help='The size of patches')
    parser.add_argument(
        '--patch-overlap-ratio',
        type=float,
        default=0.25,
        help='Ratio of overlap between two patches')
    parser.add_argument(
        '--merge-iou-thr',
        type=float,
        default=0.25,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--merge-nms-type',
        type=str,
        default='nms',
        help='NMS type for merging results')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size, must greater than or equal to 1')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Export debug results before merging')
    parser.add_argument(
        '--save-patch',
        action='store_true',
        help='Save the results of each patch. '
        'The `--debug` must be enabled.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
                                      " Can't use tta !"
        assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
                                         "in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    #  This is an mmdet issue and needs tobe fixed later.
    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    if args.deploy:
        switch_to_deploy(model)

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)
        os.mkdir(os.path.join(args.out_dir, 'temp'))

    else:
        if not os.path.exists(args.out_dir):
            os.mkdir(os.path.join(args.out_dir, 'temp'))
        else:
            for root, dirs, files in os.walk(os.path.join(args.out_dir, 'temp'), topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

    # get file list
    files, source_type = get_file_list(args.img)
    # 处理JPG情况
    if len(files) == 0:
        files = glob.glob(os.path.join(args.img,'*.JPG'))

    # start detector inference
    print(f'Performing inference on {len(files)} images.... '
          'This may take a while.')
    progress_bar = ProgressBar(len(files))

    start_time = datetime.now()

    for file in files:
        # read image
        img = mmcv.imread(file)

        # arrange slices
        height, width = img.shape[:2]
        sliced_image_object = slice_image(
            img,
            slice_height=args.patch_size,
            slice_width=args.patch_size,
            auto_slice_resolution=False,
            overlap_height_ratio=args.patch_overlap_ratio,
            overlap_width_ratio=args.patch_overlap_ratio,
        )

        # perform sliced inference
        slice_results = []
        start = 0
        while True:
            # prepare batch slices
            end = min(start + args.batch_size, len(sliced_image_object))
            images = []
            for sliced_image in sliced_image_object.images[start:end]:
                images.append(sliced_image)

            # forward the model
            slice_results.extend(inference_detector(model, images))

            if end >= len(sliced_image_object):
                break
            start += args.batch_size

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)

        img = mmcv.imconvert(img, 'bgr', 'rgb')
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        image_result = merge_results_by_nms(
            slice_results,
            sliced_image_object.starting_pixels,
            src_image_shape=(height, width),
            nms_cfg={
                'type': args.merge_nms_type,
                'iou_threshold': args.merge_iou_thr
            })

        # 保存临时结果
        with open(os.path.join(args.out_dir,'temp',filename.split('.')[0] + '.txt'), 'a') as fs:
            temp_list = []
            for i in range(len(image_result.pred_instances.bboxes)):
                labels = image_result.pred_instances.labels[i]
                scores = image_result.pred_instances.scores[i]
                bboxes = [str(int(image_result.pred_instances.bboxes[i][j]))
                          for j in range(len(image_result.pred_instances.bboxes[i]))]
                bboxes = ' '.join(bboxes)
                temp_list.append(
                    f'{labels} {scores} {bboxes} \n'
                )
            fs.writelines(temp_list)

        progress_bar.update()

    end_time = datetime.now()

    print('解析推理结果\n')

    total_list = []
    img_datetime_list = []

    for task in tqdm.tqdm(glob.glob(os.path.join(args.out_dir, 'temp', '*.txt'))):

        name = Path(task).name.split('.')[0]

        img_pathname = os.path.join(args.img, name+'.JPG')
        img_pathname = glob.glob(img_pathname)[0]
        img = cv2.imread(img_pathname)

        height, width, _ = img.shape
        xscale, yscale = calc_scalc(width, height)

        utm_center_x, utm_center_y = convert_latlon_to_utm(
            *calc_degree(img_pathname))
        utm_left_top = (utm_center_x - (width / 2) * xscale,
                        utm_center_y + (height / 2) * yscale)

        img_datetime_list.append(datetime.strptime(
            get_image_metadata(img_pathname)[306], '%Y:%m:%d %H:%M:%S'))

        with open(task) as fs:
            lines = fs.readlines()
            lines = [lines[i].split(' ') for i in range(len(lines))]
            for i in range(len(lines)):
                line = lines[i][0:-1]
                line = [float(i) for i in line]
                label = int(line[0])
                iou = float(line[1])
                pos_1 = [int(line[2]), int(line[3])]
                pos_2 = [int(line[4]), int(line[5])]

                x_center = line[2] + (line[4] - line[2])/2
                y_center = line[3] + (line[5] - line[3])/2
                x = utm_left_top[0] + int(x_center) * xscale
                y = utm_left_top[1] - int(y_center) * yscale

                # 满足约束条件
                if iou > 0.7 and label == 0:

                    obj_dic = {
                        'image': name,
                        'pos_1': ','.join([str(i) for i in pos_1]),
                        'pos_2': ','.join([str(i) for i in pos_2]),
                        'geo_x': x,
                        'geo_y': y,
                        'iou': iou,
                        'label': label,
                        'img_pathname': img_pathname
                    }

                    total_list.append(obj_dic)

    # 过滤输出结果
    total_df = pd.DataFrame(total_list)
    total_cp = total_df[total_df['iou'] > 0.85]
    index = 0

    # 去除杂株范围2米内
    d = 2

    while index <= (total_cp.shape[0] - 1):
        cell = total_cp[['geo_x', 'geo_y']].iloc[index]
        c_utm_x_c = cell['geo_x']
        c_utm_y_c = cell['geo_y']

        temp = []
        for j in range(index + 1, total_cp.shape[0]):
            line = total_cp[['geo_x', 'geo_y']].iloc[j]
            n = line.name
            c_utm_x_i = line['geo_x']
            c_utm_y_i = line['geo_y']
            if get_distance_point_to_line((c_utm_x_c, c_utm_y_c), (c_utm_x_i, c_utm_y_i)) < d:
                temp.append(n)

        total_cp = total_cp.drop(temp)
        index += 1


    flight_range = 80
    hybrid_plant_num = total_cp.shape[0]
    img_inference_num = 645
    plant_num = 50 / 0.9 * 666 * flight_range

    purity = 1 - round(hybrid_plant_num / plant_num, 8)
    purity_type = ["round_leaf", "white_flower"]
    project_name = args.project

    end_time = str(end_time.astimezone(timezone.utc))
    create_time = str(start_time.astimezone(timezone.utc))

    flight_start_date = str(min(img_datetime_list).astimezone(timezone.utc))
    flight_end_date = str(max(img_datetime_list).astimezone(timezone.utc))

    hybrid_plants = []
    label_type = ['round_leaf', 'shape_leaf']

    for name, row in total_cp.iterrows():
        id = name
        img_pathname = row['img_pathname']
        img = cv2.imread(img_pathname)
        transformer = Transformer.from_crs("epsg:32649", "epsg:4326")

        x = float(row['geo_x'])
        y = float(row['geo_y'])
        lat, lon = transformer.transform(x, y)


        hybrid_plants.append({
            "id": id,
            "type": label_type[int(row['label'])],
            "iou": row['iou'],
            "latitude": lat,
            "longitude": lon,
            "path": f"{args.out_dir}/{id}.jpg"
        })

        exp_num = 50
        pos_1 = [int(i) for i in row['pos_1'].split(',')]
        pos_2 = [int(i) for i in row['pos_2'].split(',')]


        img = cv2.rectangle(img, pos_1, pos_2, (0,0,255), 2)
        img_leaf = img[pos_1[1] - exp_num: pos_2[1] +
                       exp_num, pos_1[0] - exp_num:pos_2[0] + exp_num]
        cv2.imwrite(f'{args.out_dir}/{id}.jpg', img_leaf)

    output_dic = {
        "project_name": project_name,
        "hybrid_plant_num": hybrid_plant_num,
        "img_inference_num": img_inference_num,
        "end_time": end_time,
        "create_time": create_time,
        "purity": purity,
        "flight_start_date": flight_start_date,
        "flight_end_date": flight_end_date,
        "flight_range": flight_range,
        "purity_type": purity_type,
        "plant_num": plant_num,
        "hybrid_plants": hybrid_plants
    }

    json_str = json.dumps(output_dic)
    with open(f'{args.out_dir}/output.json', 'w') as json_file:
        json_file.write(json_str)

    print_log(f'\n结果保存到{os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
