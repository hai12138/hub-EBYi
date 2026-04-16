
from PIL import Image
import numpy as np
from sklearn.preprocessing import normalize


img_paths = "./6fc3111f7784fd41bbfca751afa672c.jpg"
img = Image.open(img_paths)
classes = ["狗", "猫", "熊猫", "蛇", "狮子", "老虎", "鸟", "人"]


from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

model = ChineseCLIPModel.from_pretrained("../../models/chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("../../models/chinese-clip-vit-base-patch16") # 预处理

img_image_feat = []
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    image_features = model.get_image_features(**inputs)
    image_features = image_features.data.numpy()
    img_image_feat.append(image_features)
img_image_feat = np.vstack(img_image_feat)
img_image_feat = normalize(img_image_feat)
print(img_image_feat.shape)

# 文本编码
img_texts_feat = []
inputs = processor(text=classes, return_tensors="pt", padding=True)
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    text_features = text_features.data.numpy()
    img_texts_feat.append(text_features)
img_texts_feat = np.vstack(img_texts_feat) # [1, 1, 512] -> [1, 512]
img_texts_feat = normalize(img_texts_feat) # 归一化期待的输入是一个二维数组
print(img_texts_feat.shape)

sim_result = np.dot(img_image_feat[0], img_texts_feat.T) # 矩阵计算
sim_idx = sim_result.argsort()[::-1][0:4]

print('图片识别结果: ', [classes[x] for x in sim_idx])