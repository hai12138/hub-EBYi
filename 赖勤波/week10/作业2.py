from openai import OpenAI
import os
import base64
from pdf2image import convert_from_path
from PIL import Image
import io

# ==================== 配置 ====================
# 1. 设置 API Key 
client = OpenAI(
    api_key="sk-14ddbf6d3e1c41c5ae4e9088e9c6dbfc",  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 本地 PDF 文件路径
pdf_path = r"G:\BaiduNetdiskDownload\week05.pdf"

# 3. 设置模型
model = "qwen-vl-plus"

POPPLER_PATH = r"G:\BaiduNetdiskDownload\Release-25.07.0-0\poppler-25.07.0\Library\bin"  

# =============================================

def pdf_first_page_to_base64(pdf_path, poppler_path=None):
    """
    将 PDF 的第一页转换为 base64 编码的 JPEG 图片
    """
    # 转换 PDF 第一页为 PIL Image
    images = convert_from_path(pdf_path, first_page=1, last_page=1, poppler_path=poppler_path)
    if not images:
        raise ValueError("无法转换 PDF 第一页")
    image = images[0]
    max_size = (1024, 1024)  
    image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # 将图像转为 base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


# 1. 将 PDF 第一页转为 base64
print("正在转换 PDF 第一页...")
try:
    img_base64 = pdf_first_page_to_base64(pdf_path, POPPLER_PATH)
    print("转换成功")
except Exception as e:
    print(f"转换失败: {e}")
    exit(1)
  
enable_thinking = False  
reasoning_content = ""
answer_content = ""
is_answering = False

completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        # 将本地图片以 data URL 形式传入
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    },
                },
                {"type": "text", "text": "请详细解析这张图片的内容，包括所有文字、表格、图表和布局结构。"},
            ],
        },
    ],
    stream=True,  # 流式输出
)

#  处理流式响应
if enable_thinking:
    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        if hasattr(chunk, 'usage') and chunk.usage:
            print("\nUsage:", chunk.usage)
        continue

    delta = chunk.choices[0].delta

    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
        print(delta.reasoning_content, end='', flush=True)
        reasoning_content += delta.reasoning_content
    elif delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "解析结果" + "=" * 20 + "\n")
            is_answering = True
        print(delta.content, end='', flush=True)
        answer_content += delta.content
