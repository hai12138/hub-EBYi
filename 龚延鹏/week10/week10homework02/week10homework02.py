from openai import OpenAI
import base64
from pdf2image import convert_from_path
import os

# ===================== 你只改这 3 个地方 =====================
PDF_PATH = r"D:\BaiduNetdiskDownload\第10周：多模态大模型\Week10-多模态大模型.pdf"          # 本地PDF
POPPLER_PATH = r"D:\BaiduNetdiskDownload\poppler\poppler-25.12.0\Library\bin"  # poppler bin路径
API_KEY = "sk-aaf17dcb8fa140ccbc455d66b2e80205"
# ==========================================================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 把 PDF 转成图片（只转前3页）
pages = convert_from_path(
    PDF_PATH,
    poppler_path=POPPLER_PATH,
    first_page=1,
    last_page=3
)

# 逐页让 Qwen-VL 识别内容
for i, img in enumerate(pages, 1):
    print(f"\n=====================================")
    print(f"正在识别 PDF 第 {i} 页...")
    print(f"=====================================\n")

    img.save("temp.jpg")
    with open("temp.jpg", "rb") as f:
        base64_img = base64.b64encode(f.read()).decode("utf-8")

    completion = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                {"type": "text", "text": "这是PDF一页，请详细描述这一页的内容、主题、文字信息"}
            ]
        }],
        stream=True
    )

    answer = ""
    for chunk in completion:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                answer += delta.content

os.remove("temp.jpg")