import cv2
import numpy as np
import datetime
import os 

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