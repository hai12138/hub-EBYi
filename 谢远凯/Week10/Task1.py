from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# 加载 CLIP 模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 加载本地小狗图片（改成你自己的图片路径）
image = Image.open("dog.jpg").convert("RGB")

# 定义候选类别（zero-shot 分类）
texts = ["a dog", "a cat", "a bird", "a car", "a tree", "a person"]

# 预处理
inputs = processor(
    text=texts,
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
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.4f}")
