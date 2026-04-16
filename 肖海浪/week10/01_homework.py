from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

# 1. 加载模型和处理器
# 使用 r"..." 原始字符串避免转义问题
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

# 1. 定义路径
raw_model_path = r"G:\Models\Chinese-clip-vit-base-patch16"

# 【关键修改 1】将路径转换为绝对路径，防止解析错误
model_path = os.path.abspath(raw_model_path)

# 【调试检查】打印路径并检查文件是否存在
print(f"正在检查路径: {model_path}")
if not os.path.exists(model_path):
    print(f"❌ 错误：找不到路径 '{model_path}'")
    exit()

# 检查 config.json 是否存在于该路径下
config_path = os.path.join(model_path, "config.json")
if not os.path.exists(config_path):
    print(f"❌ 错误：路径下缺少 config.json，请确认模型文件是否完整，或路径是否指到了子文件夹内。")
    print(f"   当前路径下的文件有: {os.listdir(model_path)}")
    exit()

print("✅ 路径检查通过，正在加载模型...")

try:
    # 【关键修改 2】强制指定 local_files_only=True
    # 这会告诉程序：只在这个文件夹里找，不要去 Hugging Face 官网比对
    model = CLIPModel.from_pretrained(model_path, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
    print("✅ 模型加载成功！")

except Exception as e:
    print(f"❌ 加载失败: {e}")
    # 如果这里依然报 Repo id 错误，说明 local_files_only 没生效或者缓存有问题
    # 尝试删除 local_files_only=True 让它尝试联网（如果你有网的话），看是否能下载
    print("提示：如果依然报 Repo id 错误，请尝试删除 local_files_only=True 参数再试一次（需联网）。")

# 后续图片处理代码...
# (保持之前的图片处理代码不变即可)
# 2. 加载本地图片
# 【建议修改】确保这里指向具体的图片文件，而不是文件夹
image_path = r"G:\八斗大模型python-AI\ai_challenger_caption_validation_20170910\dog\2.jpg"

# 简单的路径检查逻辑
if os.path.isdir(image_path):
    print(f"错误：路径 '{image_path}' 是一个文件夹，请指定具体的图片文件（如 .../dog/001.jpg）")
    exit()
elif not os.path.exists(image_path):
    print(f"错误：找不到图片文件 '{image_path}'")
    exit()

image = Image.open(image_path).convert("RGB") # 确保图片转为 RGB 格式，防止报错

# 3. 定义候选类别标签
# 注意：Chinese-CLIP 支持中文标签，你可以尝试用中文
candidate_labels = ["狗", "猫", "鸟", "花","飞机"]

# 4. 图像和文本预处理
inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)

# 5. 模型推理
print("正在计算相似度...")
with torch.no_grad():
    outputs = model(**inputs)
    # 获取图像与文本的相似度得分
    logits_per_image = outputs.logits_per_image
    # 转换为概率 (dim=1 表示对每一行的类别进行归一化)
    probs = logits_per_image.softmax(dim=1)

# 6. 输出结果
print("\n分类结果：")
# 将张量转为列表以便打印
probs_list = probs[0].tolist()
for label, prob in zip(candidate_labels, probs_list):
    print(f"{label}: {prob:.4f}")

# 找出概率最高的类别
best_label = candidate_labels[probs[0].argmax()]
print(f"\n预测结果：这是一张 {best_label} 的图片")