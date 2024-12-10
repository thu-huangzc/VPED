# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 16:30
# @Author  : thu-huangzc
# @Email   : riccardohhhhzz@gmail.com
# @File    : demo.py
# @Software: Vscode
# @Brief   : vped使用演示，支持on-line和off-line测试
# @Command : CUDA_VISIBLE_DEVICES=0 python demo.py --input ./playground/test_videos/test7.mov --event helmet --draw-results

from models.vped import VPED
import torch
import argparse

def main():
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    warnings.filterwarnings('ignore', category=UserWarning, message='torch.meshgrid')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./playground/test_videos/test3.mov', help="input video path, just write 'webcam' if you want to use webcam")
    parser.add_argument('--event', type=str, default='helmet', help='pedestrain event, currently supported event types: gender, helmet')
    parser.add_argument('--draw-results', action='store_true', help="Enable drawing results (default: True)")
    opt = parser.parse_args()

    # 运算资源选择
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    model_kwargs = {
        'yolo_detector_ckpt': './ckpt/yolo/yolov8m.pt',
        'yolo_helmet_ckpt': './ckpt/yolo/helmet_head_person_epoch10.pt',
        'clip_model_ckpt': './ckpt/clip-vit-base-patch16',
        'output_video_dir': './inference/test_videos'
    }

    model = VPED(max_duty_ratio=0.5, sample_interval=2, device=device, **model_kwargs)

    input, event, draw_results = opt.input, opt.event, opt.draw_results
    if input == 'webcam':
        model.predict(event=event, web_cam=True, draw_result=draw_results)
    else:
        model.predict(event, video_path=input, web_cam=False, draw_result=draw_results)

if __name__ == "__main__":
    main()
    