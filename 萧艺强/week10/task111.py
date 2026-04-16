import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel,ChineseCLIPProcessor, ChineseCLIPModel
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载 CLIP 模型和处理器
# 官方 openai clip 不支持中文
# https://www.modelscope.cn/models/AI-ModelScope/chinese-clip-vit-base-patch16
model = ChineseCLIPModel.from_pretrained(
    "../model/chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained(
    "../model/chinese-clip-vit-base-patch16",use_fast=False) # 预处理

# 定义候选类别标签（可根据需要扩展）
text_labels = [
    "狗", "猫", "鸟", "汽车", "桌子", "花瓶",
    "书", "电脑", "手机", "人", "动物", "家具"
]

# 加载并预处理图像
image = Image.open('../data/dog.jpg').convert("RGB")



inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

# 处理文本输入
texts = text_labels
inputs1 = processor(text=texts, return_tensors="pt", padding=True)

outputs = model.get_text_features(**inputs1)

# 获取图像特征
with torch.no_grad():
    image_outputs = model.get_image_features(**inputs)

# 计算相似度得分
logits_per_image = model.logit_scale.exp() * image_outputs @ outputs.T
probs = logits_per_image.softmax(dim=-1)[0].detach().numpy()

# 排序并输出结果
sorted_indices = probs.argsort()[::-1]
top_k = min(5, len(sorted_indices))

print("=" * 50)
print("CLIP Zero-Shot 图像分类结果")
print("=" * 50)

for i in range(top_k):
    idx = sorted_indices[i]
    label = text_labels[idx]
    confidence = probs[idx] * 100
    print(f"{i+1}. {label:<6} - 置信度: {confidence:.2f}%")

print("=" * 50)
#-----------------------
# fig, axes = plt.subplots(1, top_k + 1, figsize=(14, 4))
#
# axes[0].imshow(image)
# axes[0].set_title("原始图像")
# axes[0].axis("off")
#
# for i, idx in enumerate(sorted_indices[:top_k]):
#     axes[i + 1].axis("off")
#     ax_bar = fig.add_axes([0.73, i / top_k, 0.18, 1 / top_k])  # 调整位置
#     width = max(probs[idx], 0.01)
#     color = 'green' if i == 0 else 'grey'
#     ax_bar.barh([''], width=width, height=1, color=color)
#     ax_bar.set_xlim(0, 1)
#     ax_bar.axis('off')
#
#     ax_label = fig.add_axes([0.05, i / top_k, 0.18, 1 / top_k])
#     ax_label.text(0.1, 0.5, f"{text_labels[idx]}", fontsize=12, va='center', ha='right')
#     ax_label.axis('off')
#
# plt.tight_layout()
# plt.show()
def custom_zero_shot_classify(image_path, labels, top_k=3):
    from torch.nn.functional import softmax

    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    texts = processor(text=labels, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        text_features = model.get_text_features(**texts)
        similarity = (image_features @ text_features.T).squeeze(0)
        scores = softmax(similarity, dim=0)

    _, indices = torch.topk(scores, k=top_k)
    results = [(labels[i], scores[i].item() * 100) for i in indices.tolist()]
    return results


# 示例调用
custom_labels = ["拉布拉多", "哈士奇", "金毛", "柯基", "吉娃娃", "边境牧羊犬"]
results = custom_zero_shot_classify('../data/dog.jpg', custom_labels)

for i, (label, score) in enumerate(results, 1):
    print(f"{i}. {label:<10} - {score:.2f}%")
