# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 16:30
# @Author  : thu-huangzc
# @Email   : riccardohhhhzz@gmail.com
# @File    : utils/general.py
# @Software: Vscode
# @Brief   : 一些常用的函数工具，例如基本的文件处理、图像处理等

import cv2
import numpy as np
import datetime
import os 
from torchvision import transforms

def unified_output_video_path(input_video_path, input_dir='playground', output_dir='inference'):
    """
    统一输出视频文件格式为.mp4
    """
    video_path = (
        input_video_path.replace(input_dir, output_dir)
        .rsplit(".", 1)[0] + ".mp4"
    )
    return video_path

def webcam_output_video_path(output_dir):
    current_time = datetime.datetime.now()
    # 格式化时间为文件名
    file_name = current_time.strftime("%Y%m%d_%H%M%S")  # 格式为 YYYYMMDD_HHMMSS
    video_path = os.path.join(output_dir, f"{file_name}.mp4")
    return video_path

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 获取视频的帧率和分辨率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return frame_width, frame_height, fps, total_frames

def resize_and_pad_to_square(img, target_size=224):
    """
    对图像进行缩放，保持横纵比，将高度或宽度缩放到指定尺寸，然后填充黑边使其成为正方形。

    参数:
    img (numpy): 输入图像。
    target_size (int): 目标图像的边长，即正方形的尺寸。

    返回:
    Image: 处理后的图像对象。
    """
    # 获取图像的原始尺寸
    height, width = img.shape[:2]

    # 计算缩放比例，保持横纵比
    if width > height:
        scale = target_size / width
    else:
        scale = target_size / height

    # 计算新的宽度和高度
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩放图像
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建一个新的正方形图像，背景为黑色
    new_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # 计算粘贴的位置
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    # 将缩放后的图像粘贴到新图像的中心
    new_img[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = resized_img

    return new_img


class ResizeWithPadding(object):
    def __init__(self, target_size, fill=0):
        """
        自定义 transform，用于按比例调整大小并填充为指定尺寸
        :param target_size: 输出图像的宽高（正方形尺寸）
        :param fill: 填充颜色，默认为黑色
        """
        self.target_size = target_size
        self.fill = fill

    def __call__(self, image):
        # 获取原始宽高
        original_width, original_height = image.size
        
        # 按比例调整图像大小
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            # 宽度大于高度，调整宽度为 target_size
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            # 高度大于宽度，调整高度为 target_size
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        # 调整图像大小
        image = transforms.functional.resize(image, (new_height, new_width))
        
        # 计算填充大小
        pad_left = (self.target_size - new_width) // 2
        pad_right = self.target_size - new_width - pad_left
        pad_top = (self.target_size - new_height) // 2
        pad_bottom = self.target_size - new_height - pad_top

        # 填充图像
        image = transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        return image