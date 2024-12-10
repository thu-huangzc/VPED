import torch
from transformers import CLIPProcessor, CLIPModel
from utils.general import resize_and_pad_to_square

class CLIPClassifier(object):
    def __init__(self, clip_model_ckpt, device=None):
        """
        Initializes the CLIPClassifier with a pre-trained CLIP model.

        Args:
            model_name (str): The name of the CLIP model to use (default is "openai/clip-vit-base-patch32").
            device (str or torch.device): The device to use for computation ("cuda" or "cpu").
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载CLIP模型
        self.clip_model = CLIPModel.from_pretrained(clip_model_ckpt).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_ckpt)


    def clip_classify_withid(self, frames, texts=["man", "woman"], classes=["male", "female"]):
        # clip面对长图会采取中心裁剪方式，所以需要预先将图片处理为方形
        squared_frames = []
        for frame in frames:
            squared_frames.append(resize_and_pad_to_square(frame, 224))

        # Process frames through CLIP
        inputs = self.clip_processor(
            text=texts,
            images=squared_frames,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity logits
        probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities

        # Average probabilities over all frames
        avg_probs = probs.mean(dim=0)
        cls = avg_probs.argmax().item()
        confidence = avg_probs[cls].item()

        return (classes[cls], confidence)
    
    