import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from modelscope import snapshot_download
import numpy as np
import matplotlib.pyplot as plt



model_dir = r"D:\Application_software\Python_workspace\NLP\model"

model = ChineseCLIPModel.from_pretrained(model_dir, local_files_only=True)
processor = ChineseCLIPProcessor.from_pretrained(model_dir, local_files_only=True)

model.eval()

img_path = r"D:\Application_software\Python_workspace\NLP\Week10\dog.jpg"   # 改成你的图片绝对路径
img = Image.open(img_path).convert("RGB")

labels = [
    "小狗",
    "猫",
    "鸟",
    "汽车",
    "房子",
    "人",
    "飞机",
    "船"
]

texts = [f"这是一张{label}的图片" for label in labels]

image_inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

text_inputs = processor(text=texts, return_tensors="pt", padding=True)

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

similarity = (image_features @ text_features.T).squeeze(0)   # [num_labels]
probs = similarity.softmax(dim=0).cpu().numpy()

pred_idx = int(np.argmax(probs))

print("图像分类结果：", labels[pred_idx])
print("\n各类别概率：")
for label, prob in zip(labels, probs):
    print(f"{label}: {prob:.4f}")


plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(f"预测结果: {labels[pred_idx]}")
plt.xticks([])
plt.yticks([])
plt.show()
