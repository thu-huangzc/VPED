# -*- coding: utf-8 -*-
# @Time    : 2024/12/09 13:30
# @Author  : thu-huangzc
# @Email   : riccardohhhhzz@gmail.com
# @File    : train_yolo.py
# @Software: Vscode
# @Brief   : 使用自制数据集对yolov8进行训练
# @Command : CUDA_VISIBLE_DEVICES=0,1 python train_yolo.py --epoch 100

from models.yolo import load_yolo
import argparse
import torch

def train():
    weights, dataset, imgsz, epoch = opt.weights, opt.dataset, opt.img_size, opt.epoch

    # load pretrained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_yolo(weights, device)

    # train model
    results = model.train(data=dataset, epochs=epoch, imgsz=imgsz)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./ckpt/yolo/yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--dataset', type=str, default='./datasets/Safety_Helmet_Train_dataset/data.yaml', help='yolo dataset yaml')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--epoch', type=int, default=100, help='train epoch')
    opt = parser.parse_args()

    train()
