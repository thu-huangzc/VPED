# -*- coding: utf-8 -*-
# @Time    : 2024/12/17 22:30
# @Author  : thu-huangzc
# @Email   : riccardohhhhzz@gmail.com
# @File    : models/classifier/phone_detector.py
# @Software: Vscode
# @Brief   : 打电话检测器（只判断、不定位）

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from utils.general import ResizeWithPadding

class PhoneDetector(object):
    def __init__(self, model_path, task="phone", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.task = task
        self.labels = {0: "normal", 1: task}

        # Image preprocessing (same as used during training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            ResizeWithPadding(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_model(self):
        model = models.efficientnet_b4(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 2),
        )
        return model

    def preprocess_batch(self, images):
        processed_images = []
        for img in images:
            if img.shape[-1] == 3:  # Ensure the image has 3 channels (H, W, C)
                img = self.transform(img)
                processed_images.append(img)
            else:
                raise ValueError("Each image must have 3 channels (H, W, C).")

        return torch.stack(processed_images).to(self.device)

    def predict(self, images):
        # Preprocess the input batch
        inputs = self.preprocess_batch(images)  # Shape: (batch_size, 3, 224, 224)

        with torch.no_grad():
            outputs = self.model(inputs)  # Raw logits
            probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
            confidences, predicted = torch.max(probabilities, 1)

        return [(self.labels[p.item()], c.item()) for p, c in zip(predicted, confidences)]

if __name__ == "__main__":
    model_path = "ckpt/classifier/phone_detection.pth"

    # Initialize detector
    detector = PhoneDetector(model_path, 'phone')

    # Example batch of images (numpy array of shape [batch_size, H, W, C])
    # Replace with actual numpy images
    example_images = np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8)  # Batch of 4 random images

    # Perform prediction
    predictions = detector.predict(example_images)
    print(f"Predictions: {predictions}")
