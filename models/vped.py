# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 16:30
# @Author  : thu-huangzc
# @Email   : riccardohhhhzz@gmail.com
# @File    : models/vped.py
# @Software: Vscode
# @Brief   : Video Pedestrian Event Detection(VPED)，给定一段视频以及待检测的事件，识别检测并跟踪视频中的人物，并进行相应的事件判断

import cv2
import torch
from models.classifier.clip_classifier import CLIPClassifier
from models.yolo import load_yolo
import paddlehub as hub
from models.classifier.phone_detector import PhoneDetector
from utils.general import *
from tqdm import tqdm

class VPED(object):
    def __init__(self, max_duty_ratio=0.5, sample_interval=2, device=None, **kwargs):
        # 选择运行环境
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 获取模型权重文件
        self.yolo_detector_ckpt = kwargs.get('yolo_detector_ckpt', './ckpt/yolo/yolov8m.pt') # 使用预训练yolov8+bytetracker对行人进行目标检测追踪
        self.yolo_helmet_ckpt = kwargs.get('yolo_helmet_ckpt', './ckpt/yolo/helmet_head_person_epoch10.pt') # 使用在自制数据集上训练后的yolov8作为安全帽识别和检测模型
        self.clip_model_ckpt = kwargs.get('clip_model_ckpt', './ckpt/clip-vit-base-patch16') # 使用clip基于人体图像进行性别识别
        self.yolo_cigarette_ckpt = kwargs.get('yolo_cigarette_ckpt', './ckpt/yolo/cigarette_epoch20.pt') # 使用在自制数据集上训练后的yolov8作为香烟检测模型
        self.phone_detector_ckpt = kwargs.get('phone_detector_ckpt', './ckpt/classifier/phone_detection.pth') # 使用efficientnet对打电话进行识别

        # 其他参数
        self.max_duty_ratio = max_duty_ratio
        self.sample_interval = sample_interval
        self.support_events = set(['gender', 'helmet', 'mask', 'smoking', 'phone'])
        self.output_video_dir = kwargs.get('output_video_dir', './inference/test_videos')

        # 加载模型
        self.det = load_yolo(self.yolo_detector_ckpt, self.device)
        self.helmet_det = load_yolo(self.yolo_helmet_ckpt, self.device)
        self.gender_classifer = CLIPClassifier(self.clip_model_ckpt, self.device)
        self.mask_det = hub.Module(name="pyramidbox_lite_mobile_mask")
        self.cigarette_det = load_yolo(self.yolo_cigarette_ckpt, self.device)
        self.phone_det = PhoneDetector(self.phone_detector_ckpt, task='phone', device=self.device)

        # 目标追踪帧缓存
        self.id_frames = dict()

        # 追踪目标身份信息列表
        self.id_forms = dict()

    def predict(self, event='helmet', video_path=None, web_cam=False, draw_result=True):
        if event not in self.support_events:
            raise NotImplementedError(f"The event '{event}' is currently not supported. Currently supported events are {self.support_events}")

        # 确定输入来源是本地视频还是摄像头
        if web_cam:
            source = 0
        else:
            # 获取输入视频的信息
            source = video_path
            frame_width, frame_height, fps, frame_nums = get_video_info(video_path)
            sample_fps = int(self.max_duty_ratio * fps)

        if draw_result:
            # 定义输出视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if web_cam:
                output_video_path = webcam_output_video_path(self.output_video_dir)
                out = cv2.VideoWriter(output_video_path, fourcc, 30, (1280, 720)) # 这里设置为摄像头的规格
            else:
                output_video_path = unified_output_video_path(video_path)
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # 对行人进行实时目标检测追踪，track_results不是list，而是generator，数据是动态生成的
        track_results = self.det.track(source, classes=[0], stream=True, verbose=False)
        cur_frame = 0
        last_sample_frame = cur_frame - self.sample_interval

        if web_cam:
            for res in track_results:
                frame_img = res.orig_img
                frame_bboxes = res.boxes
                # 采集视频帧存入到id_frames里
                if (cur_frame % fps < min(sample_fps, 24) and cur_frame - last_sample_frame == self.sample_interval) or cur_frame % fps == 0:
                    last_sample_frame = cur_frame
                    self._update_id_frames(frame_img, frame_bboxes)

                # 采集结束，进行二阶段行为属性识别（性别、安全帽、抽烟等）
                if cur_frame % fps == min(sample_fps, 24):
                    self._update_id_forms(event)
                    self._clear_id_frames()

                cur_frame += 1

                if draw_result: # 绘制bbox以及属性识别结果到新视频中
                    drawed_frame = self._draw_one_frame(frame_img, frame_bboxes, event=event)
                    out.write(drawed_frame)
        else:
            with tqdm(total=frame_nums, desc="Running") as pbar:
                for res in track_results:
                    frame_img = res.orig_img
                    frame_bboxes = res.boxes
                    # 采集视频帧存入到id_frames里
                    if (cur_frame % fps < min(sample_fps, 24) and cur_frame - last_sample_frame == self.sample_interval) or cur_frame % fps == 0:
                        last_sample_frame = cur_frame
                        self._update_id_frames(frame_img, frame_bboxes)

                    # 采集结束，进行二阶段行为属性识别（性别、安全帽、抽烟等）
                    if cur_frame % fps == min(sample_fps, 24):
                        self._update_id_forms(event)
                        self._clear_id_frames()

                    cur_frame += 1

                    if draw_result: # 绘制bbox以及属性识别结果到新视频中
                        drawed_frame = self._draw_one_frame(frame_img, frame_bboxes, event=event)
                        out.write(drawed_frame)

                    pbar.update(1)
        
        print(self.id_forms)
        if draw_result:
            out.release()
            cv2.destroyAllWindows()
        
        return self.id_forms
                    
    def _update_id_frames(self, img, bboxes):
        for obj in bboxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = obj.xyxy[0].cpu().numpy().astype(int)  # 转换为 numpy 数组并解包
            # 获取追踪 ID
            track_id = obj.id[0].item() if obj.id is not None else None
            if track_id in self.id_frames:
                crop_img = img[y1:y2, x1:x2, :]
                self.id_frames[track_id].append(crop_img)
            else:
                self.id_frames[track_id] = []
    
    def _clear_id_frames(self):
        self.id_frames = dict()
    
    def _update_id_forms(self, event='all'):
        for id in self.id_frames:
            if len(self.id_frames[id]) > 0:
                # 安全帽及人头检测，原则为多数投票表决；同时也进一步判断目标是否是人
                helmet, conf_helmet = self._detect_helmet(id)
                is_human = helmet == 'head' or helmet == 'helmet'
                if is_human:
                    if event == 'all' or event == 'helmet':
                        self._update_attr((helmet, conf_helmet), id, 'helmet')
                    if event == 'all' or event == 'gender':
                        gender, conf_gender = self._detect_gender(id) # 性别识别
                        self._update_attr((gender, conf_gender), id, 'gender')
                    if event == 'all' or event == 'mask':
                        mask, conf_mask = self._detect_mask(id) # 口罩识别
                        self._update_attr((mask, conf_mask), id, 'mask')
                    if event == 'all' or event == 'smoking':
                        smoking, conf_smoking = self._detect_smoking(id)
                        self._update_attr((smoking, conf_smoking), id, 'smoking')
                    if event == 'all' or event == 'phone':
                        phone, conf_phone = self._detect_phone(id)
                        self._update_attr((phone, conf_phone), id, 'phone')
                    

    def _update_attr(self, result, id, attr):
        cls, conf = result
        if id not in self.id_forms:
            self.id_forms[id] = dict()
        if attr not in self.id_forms[id]:
            self.id_forms[id][attr] = (None, 0.0)
        # 最大置信度覆盖更新策略
        if cls != self.id_forms[id][attr][0]:
            self.id_forms[id][attr] = (cls, conf)
        elif self.id_forms[id][attr][1] < conf:
            self.id_forms[id][attr] = (cls, conf)
            
    def _detect_helmet(self, id):
        """
        检测是否佩戴安全帽
        """
        classes = [None, 'head', 'helmet']
        helmet_results = self.helmet_det(self.id_frames[id], classes=[1,2], verbose=False)
        # 用于存储每一帧的检测结果，-1表示没有检测到，其余为检测到佩戴安全帽的帧平均概率值
        frame_results = []
        for res in helmet_results:
            if len(res.boxes) == 0:  # 如果没有检测到
                frame_results.append(-1)
            else:
                helmet_probs = [box.conf.item() for box in res.boxes if box.cls == 2]  # cls==2 表示 helmet
                if helmet_probs:
                    frame_results.append(sum(helmet_probs) / len(helmet_probs))  # 计算平均概率
                else:
                    head_probs = [box.conf.item() for box in res.boxes if box.cls == 1]  # cls==1 表示 head
                    if head_probs:
                        frame_results.append(1 - sum(head_probs) / len(head_probs))
            
        # 如果没有检测到的次数超过帧数一半，认定为不是真人
        no_detection_count = frame_results.count(-1)
        no_detection_ratio = no_detection_count / len(frame_results)
        if no_detection_ratio > 0.5:
            final_prediction = 0
            conf = no_detection_ratio
        else: # 否则，将非-1的概率值取平均后，如果超过0.5，认定为佩戴了安全帽，否则认定为没有佩戴
            valid_probs = [prob for prob in frame_results if prob != -1]
            helmet_prob = sum(valid_probs) / len(valid_probs)
            if helmet_prob > 0.5:
                final_prediction = 2
                conf = helmet_prob
            else:
                final_prediction = 1
                conf = 1 - helmet_prob

        return classes[final_prediction], conf
    
    def _detect_gender(self, id):
        """
        识别性别
        """
        frames = self.id_frames[id]
        return self.gender_classifer.clip_classify_withid(
            frames, 
            texts=["A man's body", "A woman's body"], 
            classes=["male", "female"])

    def _detect_mask(self, id):
        """
        检测是否佩戴口罩
        """
        classes = ["no mask", "mask"]
        frames = self.id_frames[id]
        results = self.mask_det.face_detection(images=frames)
        score = 0 #佩戴口罩的概率
        for res in results:
            label, conf = res['data'][0]['label'].lower(), res['data'][0]['confidence']
            if label == 'mask':
                score += conf
            else:
                score += 1 - conf
        score /= len(results)
        if score > 0.5: 
            final_prediction = 1
            conf = score
        else: 
            final_prediction = 0
            conf = 1 - score
        
        return classes[final_prediction], conf
     
    def _detect_smoking(self, id):
        """
        检测是否吸烟
        """
        classes = ['no smoking', 'smoking']
        ciga_results = self.cigarette_det(self.id_frames[id], classes=[0], verbose=False)
        # 用于存储每一帧的检测结果，-1表示没有检测到，其余为检测到香烟的帧平均概率值
        frame_results = []
        for res in ciga_results:
            if len(res.boxes) == 0:  # 如果没有检测到
                frame_results.append(-1)
            else:
                ciga_probs = [box.conf.item() for box in res.boxes if box.cls == 0]  # cls==2 表示 helmet
                if ciga_probs:
                    frame_results.append(sum(ciga_probs) / len(ciga_probs))  # 计算平均概率
            
        # 如果没有检测到的次数超过阈值，认定为没有抽烟
        no_detection_count = frame_results.count(-1)
        no_detection_ratio = no_detection_count / len(frame_results)
        if no_detection_ratio > 0.6:
            final_prediction = 0
            conf = no_detection_ratio
        else: # 否则，将非-1的概率值取平均作为抽烟的概率值
            valid_probs = [prob for prob in frame_results if prob != -1]
            ciga_prob = sum(valid_probs) / len(valid_probs)
            final_prediction = 1
            conf = ciga_prob

        return classes[final_prediction], conf

    def _detect_phone(self, id):
        """
        检测是否打电话
        """
        classes = ["no phone", "phone"]
        frames = self.id_frames[id]
        results = self.phone_det.predict(frames)
        score = 0 # 打电话的概率
        for res in results:
            label, conf = res
            if label == 'phone': score += conf
            else: score += 1-conf
        score /= len(results)
        if score > 0.5: 
            final_prediction = 1
            conf = score
        else: 
            final_prediction = 0
            conf = 1 - score
        
        return classes[final_prediction], conf
        
    def _draw_one_frame(self, frame, bboxes, event='gender', line_thickness=2, fontsize=2):
        classes_index = {'helmet':0, 'head':1, 'male': 0, 'female': 1, 'mask':0, 'no mask': 1, 'no smoking': 0, 'smoking': 1, 'no phone': 0, 'phone': 1}
        new_frame = frame.copy()
        bboxes2draw = []
        for obj in bboxes:
            x1, y1, x2, y2 = obj.xyxy[0].cpu().numpy()
            track_id = obj.id[0].item() if obj.id is not None else None
            if track_id in self.id_forms:
                cls, conf = self.id_forms[track_id][event]
                bboxes2draw.append(
                    (x1, y1, x2, y2, cls, conf, track_id)
                )
            elif track_id:
                bboxes2draw.append(
                    (x1, y1, x2, y2, '', 1.0, track_id)
                )
        for (x1, y1, x2, y2, cls, conf, id) in bboxes2draw:
            if cls == '' or cls == None: continue
            color = (0,255,0)
            if classes_index[cls] == 1:
                color = (0,0,255)
            c1,c2 = (int(x1),int(y1)), (int(x2),int(y2))
            cv2.rectangle(new_frame, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
            text = 'id:{},{},{:.2f}'.format(int(id), cls, conf)
            t_size = cv2.getTextSize(text, 0, fontScale=fontsize / 3, thickness=fontsize)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(new_frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(new_frame, text, (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=fontsize, lineType=cv2.LINE_AA)
        return new_frame