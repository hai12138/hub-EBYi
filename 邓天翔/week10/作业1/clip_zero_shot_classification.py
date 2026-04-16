# CLIP零样本图像分类示例
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# 加载预训练的中文CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChineseCLIPModel.from_pretrained("../model/AI-ModelScope/chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("../model/AI-ModelScope/chinese-clip-vit-base-patch16") # 预处理

# 加载并预处理图像
image_path = "pic/dog.png"
image = Image.open(image_path)

# 定义待分类的类别
categories = [
    "一张狗的照片",
    "一张猫的照片",
    "一张鸟的照片"
]

# 处理图像和文本
inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 将类别转换为文本编码
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 打印预测结果
print(f"图像: {image_path}")
print("\n分类概率:")
for i, category in enumerate(categories):
    print(f"{category}: {probs[0][i]:.4f}")

# 获取最高概率的类别
predicted_idx = probs.argmax()
print(f"\n预测的类别: {categories[predicted_idx]}")
print(f"置信度: {probs[0][predicted_idx]:.4f}")