# -*- coding: utf-8 -*-
# @Time    : 2024/12/09 13:30
# @Author  : thu-huangzc
# @Email   : riccardohhhhzz@gmail.com
# @File    : models/yolo.py
# @Software: Vscode
# @Brief   : 加载yolov8模型，测试yolov8及其自带追踪算法效果
# @Command : CUDA_VISIBLE_DEVICES=6 python models/yolo.py --weights ./ckpt/yolo/yolov8m.pt --input ./playground/test_videos/test2.mov

from ultralytics import YOLO

def load_yolo(model_path, device):
    model = YOLO(model_path).to(device)
    return model

if __name__ == "__main__":
    import torch
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./ckpt/yolo/helmet_head_person_epoch10.pt', help='model.pt path(s)')
    parser.add_argument('--input', type=str, default='./playground/test_videos/test3.mov', help='input video path')
    opt = parser.parse_args()

    model_path, input_video_path = opt.weights, opt.input
    print(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_yolo(model_path, device)

    input_video_path = opt.input
    output_video_path = input_video_path.replace('playground', 'inference')
    output_video_path = output_video_path.replace('.mov', '.mp4')
    
    start_time = time.time()  # 记录起始时间

    results = model.track(input_video_path, classes=[0], stream=True, verbose=False)

    import cv2

    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 获取视频的帧率和分辨率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    # 定义输出视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for result in results:
        frame = result.orig_img
        frame_with_bbox = result.plot(img=frame, line_width=1, font_size=1)
        out.write(frame_with_bbox)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"运行时间: {elapsed_time:.4f} 秒")

    # 释放资源
    out.release()
    cv2.destroyAllWindows()
    