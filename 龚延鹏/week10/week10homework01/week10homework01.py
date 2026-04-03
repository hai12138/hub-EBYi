from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# 加载 CLIP 模型和处理器
model_name = r"D:\BaiduNetdiskDownload\models\openai\clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# ===================== 替换成你的本地图片路径 =====================
image_path = r"D:/HuaweiMoveData/Users/27999/Pictures/nlp_study/R-C.jpg"  # 你的小狗图片
# ==================================================================

image = Image.open(image_path).convert("RGB")

# 定义你想让模型“猜测”的类别（零样本，无需训练）
candidate_labels = [
    "小狗",
    "小猫",
    "小鸟",
    "汽车",
    "树木",
    "花",
    "人",
    "食物"
]

# 预处理图像和文本
inputs = processor(
    text=candidate_labels,
    images=image,
    return_tensors="pt",
    padding=True
)

# 推理
with torch.no_grad():
    outputs = model(**inputs)

# 计算相似度并取概率
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# 输出结果
print("CLIP 零样本图像分类结果：")
for label, prob in zip(candidate_labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")

# 找出最可能的类别
max_prob, max_idx = torch.max(probs, dim=1)
print("\n预测结果：", candidate_labels[max_idx])