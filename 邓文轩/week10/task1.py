import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model_path = r"D:\models\AI-ModelScope\chinese-clip-vit-base-patch16"

print("正在加载模型和处理器...")
model = ChineseCLIPModel.from_pretrained(model_path)
processor = ChineseCLIPProcessor.from_pretrained(model_path)

image_path = "dog.jpg"

if not os.path.exists(image_path):
    print(f"请确保当前目录下存在 {image_path} 图片文件")
    exit(1)

image = Image.open(image_path).convert("RGB")

candidate_labels = ["狗", "猫", "鸟", "鱼", "兔子", "老虎", "狮子", "大象"]

print(f"正在对图片进行 zero-shot 分类...")
print(f"候选标签: {candidate_labels}")
print("-" * 50)

image_inputs = processor(images=image, return_tensors="pt")

text_inputs = processor(text=candidate_labels, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**image_inputs, **text_inputs)
    
    image_features = outputs.image_embeds
    image_features = F.normalize(image_features, dim=-1)
    
    text_features = outputs.text_embeds
    text_features = F.normalize(text_features, dim=-1)
    
    similarities = (image_features @ text_features.T).squeeze(0)
    
    probs = F.softmax(similarities * 100, dim=-1)

results = []
for i, label in enumerate(candidate_labels):
    results.append({
        "label": label,
        "score": probs[i].item()
    })

results.sort(key=lambda x: x["score"], reverse=True)

print("分类结果:")
for item in results:
    print(f"  {item['label']}: {item['score']:.4f}")

print("-" * 50)
best_match = results[0]
print(f"最可能的类别: {best_match['label']} (置信度: {best_match['score']:.4f})")
