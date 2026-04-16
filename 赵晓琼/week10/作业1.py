'''
对小狗图片进行clip的zero shot classification图像分类（零样本分类）
CLIP模型是一种基于对比学习的多模态模型，它利用的假设是，在学习到的嵌入空间中，相似的实例应靠的更近，而不相似的实例离得更远。
CLIP核心思想：使用海量的弱监督文本通过对比学习，将图像和文本映射到一个共享的向量空间，理解图像与文本之间的语义关系。
'''
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import matplotlib.pyplot as plt

# Mac 系统常用的中文字体设置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# 加载模型和处理器
model = ChineseCLIPModel.from_pretrained('../../../AI-ModelScope/chinese-clip-vit-base-patch16')
processor = ChineseCLIPProcessor.from_pretrained('../../../AI-ModelScope/chinese-clip-vit-base-patch16')

# 加载图片
image = Image.open('./dog.jpeg').convert('RGB')

# 定义候选类别（可以任意扩展）
candidate_labels = [
    "一张狗的照片",           
    "一只小狗狗趴在草地上",          
    "一只鸟的照片",          
    "一只猫的照片",           
    "一只可爱的小狗狗",         
    "一个高高的人",       
    "德国牧羊犬",              
    "贵宾犬",         
    "美丽的风景",    
    "一只可爱的小狗狗吐着舌头趴在草地上",          
]
 
# 处理输入
# processor会自动处理文本的分词和图片的缩放/归一化，并转换为PyTorch Tensor
inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

print(inputs)

# 推理
# 不需要计算梯度，加快速度
with torch.no_grad():
    outputs = model(**inputs) # 模型计算图片向量和所有文本向量之间的余弦相似度
    ''' 
    **inputs字典解包
    inputs = {
        "text": ["a photo of a cat", "a photo of a dog"],
        "images": <PIL图像对象>,
        "padding": True
    }
    解包后：model(text=["a photo of a cat", ...], images=<PIL图像对象>, padding=True)
    '''

# 获取结果
# logits_per_image 表示图片和文本的相似度矩阵
# 对其做softmax归一化，将其转换为概率
probs = outputs.logits_per_image.softmax(dim=1)

# 结果
print('候选项：', candidate_labels)
print('概率预测：', probs)

print('\n-----分类结果------')
results = []
for label, prob in zip(candidate_labels, probs[0]): # zip将类别名称和概率值一一对应捆绑在一起，方便同时遍历
    results.append({
        "label": label,
        "probability": prob.item()
    })
    print(f"{label}: {prob.item()*100:.2f}%")


# 可视化
def visualize_result(image_path, results):
    """可视化分类结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示图片
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.set_title("输入图片")
    ax1.axis("off")
    
    # 显示分类结果
    labels = [r["label"] for r in results[:]]
    probs = [r["probability"] for r in results[:]]
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
    bars = ax2.barh(labels, probs, color=colors)
    ax2.set_xlabel("概率")
    ax2.set_title("Zero-shot 分类结果")
    
    # 添加概率标签
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{prob:.2%}", va='center')
    
    plt.tight_layout()
    plt.show()

# 可视化
visualize_result('./dog.jpeg', results)