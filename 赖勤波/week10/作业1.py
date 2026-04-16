from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import clip


# 官方 openai clip 不支持中文
# https://www.modelscope.cn/models/AI-ModelScope/chinese-clip-vit-base-patch16
model = ChineseCLIPModel.from_pretrained("G:/BaiduNetdiskDownload/models/chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("G:/BaiduNetdiskDownload/models/chinese-clip-vit-base-patch16") # 预处理

# 定义候选类别（中文）
class_names = [
    "一只狗",      # 狗
    "一只猫",      # 猫
    "一只鸟",      # 鸟
    "一辆汽车",    # 汽车
    "一栋房子",    # 房子
    "一朵花",      # 花
    "一个人",      # 人
    "一辆自行车"   # 自行车
]

# 加载并预处理本地图片
image_path = "G:/BaiduNetdiskDownload/dog.jpg" 
imgs = Image.open(image_path)
inputs = processor(images=imgs, return_tensors="pt")
with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

# 替换原来的 text_features 计算部分
inputs_text = processor(text=class_names, padding=True, return_tensors="pt")
with torch.no_grad():
    # 通过文本模型获取输出
    text_outputs = model.text_model(**inputs_text)
    pooled_output = text_outputs.pooler_output   # 这是 [batch, hidden_size]
    # 如果 pooled_output 为 None，则尝试使用 last_hidden_state 的 CLS token
    if pooled_output is None:
        # 取 last_hidden_state 的第一个 token（[CLS]）
        last_hidden = text_outputs.last_hidden_state
        pooled_output = last_hidden[:, 0, :]
    if hasattr(model, 'text_projection') and model.text_projection is not None:
        text_features = model.text_projection(pooled_output)
    else:
        text_features = pooled_output
    
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# 计算相似度并得到概率
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
probs = similarity.cpu().numpy()[0]

# 输出预测结果（按概率降序）
sorted_indices = probs.argsort()[::-1]
print("Top-5 预测结果：")
for i, idx in enumerate(sorted_indices[:5]):
    print(f"{i+1}. {class_names[idx]}: {probs[idx]:.4f}")

"""
运行结果：
Top-5 预测结果：
1. 一只狗: 0.8943
2. 一朵花: 0.0760
3. 一只猫: 0.0204
4. 一只鸟: 0.0060
5. 一个人: 0.0021
"""
