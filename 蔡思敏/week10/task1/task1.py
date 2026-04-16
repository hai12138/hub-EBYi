"""
@Author  :  CAISIMIN
@Date    :  2026/3/29 20:39
"""

from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from PIL import Image
import torch
from sklearn.preprocessing import normalize

# 标签类别
labels = ["小狗", "猫咪"]
model = ChineseCLIPModel.from_pretrained("../../../models/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("../../../models/chinese-clip-vit-base-patch16")

img = Image.open('img.png')
img_input = processor(images=img, return_tensors="pt")
text_input = processor(text=labels, return_tensors="pt")

with torch.no_grad():
    img_features = model.get_image_features(**img_input)
    text_features = model.get_text_features(**text_input)

    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sim = (img_features@text_features.T).softmax(dim=1)

    pred = labels[sim.argmax().item()]
    print(f"图片分类预测结果为：{pred}") # 小狗


