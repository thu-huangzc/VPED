# -*- coding: utf-8 -*-
# @Time    : 2024/12/09 13:30
# @Author  : thu-huangzc
# @Email   : riccardohhhhzz@gmail.com
# @File    : datasets/detect_person.py
# @Software: Vscode
# @Brief   : 使用yolov8检测SHWD中的人体，并保存结果
# @Command : CUDA_VISIBLE_DEVICES=0 python -m datasets.detect_person

import os
from models.yolo import load_yolo
import torch
from tqdm import tqdm
import shutil

# 加载模型（请替换为你实际使用的权重文件路径）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_yolo('ckpt/yolo/yolov8m.pt', device)  # 或者使用其他YOLOv8模型权重

# 数据路径和输出路径
image_dir = 'datasets/VOC2028/JPEGImages'
output_dir = 'datasets/VOC2028/PersonPseudoLabels'

# 确保输出目录存在
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # delete output folder
os.makedirs(output_dir)  # make new output folder

# 遍历所有图像
for image_name in tqdm(os.listdir(image_dir)):
    image_path = os.path.join(image_dir, image_name)

    # 检测图像
    results = model(image_path, verbose=False)

    # 获取检测结果
    detections = results[0].boxes.data  # tensor: [x_min, y_min, x_max, y_max, confidence, class]

    # 过滤出person类别
    person_detections = detections[detections[:, 5] == 0]

    # YOLO格式转换：从(x_min, y_min, x_max, y_max)转换为(x_center, y_center, width, height)
    yolo_format = []
    for det in person_detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        x_center = (x_min + x_max) / 2 / results[0].orig_shape[1]
        y_center = (y_min + y_max) / 2 / results[0].orig_shape[0]
        width = (x_max - x_min) / results[0].orig_shape[1]
        height = (y_max - y_min) / results[0].orig_shape[0]
        yolo_format.append(f"{int(cls)} {x_center} {y_center} {width} {height}")

    # 保存结果到文件
    output_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.txt')
    with open(output_path, 'w') as f:
        f.write("\n".join(yolo_format))

print("检测完成，结果已保存到", output_dir)