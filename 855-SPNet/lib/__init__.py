


import cv2
import numpy as np

from PIL import Image
import xmltodict
from pyproj import Proj
from math import sin, cos

import json
import datetime
import time


from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_image_metadata(image_path):
    """Retrieve metadata from an image."""
    image = Image.open(image_path)
    
    # Extract the exif data (if it exists)
    image_info = image._getexif()
    # if image_info is not None:
    #     for tag, value in image_info.items():
    #         tag_name = TAGS.get(tag, tag)
    #         if tag_name == "GPSInfo":
    #             for t in value:
    #                 sub_tag_name = GPSTAGS.get(t, t)
    #                 image_info[sub_tag_name] = value[t]
    #             image_info.pop(tag_name)
    
    return image_info



def time_to_utc(year,month,day,h,m,s):
    timestamp = datetime.datetime(year, month, day, h, m, s).replace(tzinfo=datetime.timezone.utc).timestamp()
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    return str(dt)



def get_distance_point_to_line(point1, point2):
    import math
    x0, y0 = point1
    x1, y1 = point2

    # Calculate the distance
    d = math.sqrt((x1-x0)**2 + (y1-y0)**2)

    return d


def convert_latlon_to_utm(latitude, longitude):
    utm_zone = int((longitude + 180) // 6) + 1  # 计算UTM带号

    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=True)
    utm_easting, utm_northing = utm_proj(longitude, latitude)

    return utm_easting, utm_northing


def calc_degree(img_pathname):
    def dfm_to_degree(dfm_tuple):

        dfm_list = list(dfm_tuple)
        dfm_list = [float(i) for i in dfm_list]
        output = 0
        output += dfm_list[0]
        output += dfm_list[1] / 60
        output += dfm_list[2] / 60 / 60

        return output

    img = Image.open(img_pathname)
    info = img._getexif()

    N = info[34853][2]
    E = info[34853][4]
    H = info[34853][6]

    N_d = dfm_to_degree(N)
    E_d = dfm_to_degree(E)
    return N_d, E_d


def calc_scalc(image_width, image_height):
    # 相机参数
    focal_length = 35  # 焦距为35毫米
    sensor_width = 35.9  # 传感器宽度为24毫米
    sensor_height = 24  # 传感器高度为18毫米

    # 计算视野范围
    distance = 15  # 相机到被拍摄物体的距离

    # 计算比例尺
    xscale = (sensor_width * distance) / (image_width * focal_length)
    yscale = (sensor_height * distance) / (image_height * focal_length)

    return xscale, yscale


def getFlightYawDegree(img_pathname):
    with open(img_pathname, 'rb') as f:
        f.seek(0)
        byte_data = f.read(100000)

    text = ''

    for b in byte_data:
        try:
            text += b.to_bytes(1, 'big').decode('utf-8')
        except:
            pass

    text = text[text.find('<?xpacket'):text.find(
        '</x:xmpmeta>')+len('</x:xmpmeta>')]
    data = xmltodict.parse(text)
    FlightYawDegree = data['x:xmpmeta']['rdf:RDF']['rdf:Description']['@drone-dji:FlightYawDegree']

    return FlightYawDegree


def Projection2ImageRowCol(adfGeoTransform, dProjX, dProjY):

    dTemp = adfGeoTransform[1] * adfGeoTransform[5] - \
        adfGeoTransform[2] * adfGeoTransform[4]

    dCol = (adfGeoTransform[5] * (dProjX - adfGeoTransform[0]) -
            adfGeoTransform[2] * (dProjY - adfGeoTransform[3])) / dTemp + 0.5

    dRow = (adfGeoTransform[1] * (dProjY - adfGeoTransform[3]) -
            adfGeoTransform[4] * (dProjX - adfGeoTransform[0])) / dTemp + 0.5

    return [int(dCol), int(dRow)]


def calc_yaw_x_y(yaw_world, dProjX, dProjY):
    # 顺时针针转
    R = [[cos(yaw_world), sin(yaw_world)],
         [sin(yaw_world), cos(yaw_world)]]
    x_new = R[0][0] * dProjX + R[0][1] * dProjY
    y_new = - R[1][0] * dProjX + R[1][1] * dProjY
    return x_new, y_new


def rotate_image(image, angle):
    errors = -3.5

    # 获取图像尺寸
    height, width = image.shape[:2]

    # 设置旋转中心
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle + errors, 1)

    # 计算旋转后的新尺寸
    radians = np.radians(angle)
    sin = np.sin(radians)
    cos = np.cos(radians)
    new_width = int((height * np.abs(sin)) + (width * np.abs(cos)))
    new_height = int((height * np.abs(cos)) + (width * np.abs(sin)))

    # 更新旋转矩阵的平移量
    rotation_matrix[0, 2] += (new_width // 2) - center[0]
    rotation_matrix[1, 2] += (new_height // 2) - center[1]

    # 进行仿射变换
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height))

    return rotated_image




