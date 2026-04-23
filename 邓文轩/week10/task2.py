import os
import base64
import fitz
import dashscope
from dashscope import MultiModalConversation

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

if not dashscope.api_key:
    print("错误: 请设置系统环境变量 DASHSCOPE_API_KEY")
    exit(1)

pdf_path = "sample.pdf"

if not os.path.exists(pdf_path):
    print(f"请确保当前目录下存在 {pdf_path} PDF文件")
    exit(1)

print(f"正在转换PDF第一页为图片...")
pdf_document = fitz.open(pdf_path)
first_page = pdf_document[0]

zoom = 2
mat = fitz.Matrix(zoom, zoom)
pix = first_page.get_pixmap(matrix=mat)

temp_image_path = "temp_pdf_page.jpg"
pix.save(temp_image_path)
pdf_document.close()
print(f"PDF第一页已保存为: {temp_image_path}")

print("正在使用 Qwen-VL 解析PDF内容...")
print("-" * 50)

with open(temp_image_path, "rb") as f:
    image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {"image": f"data:image/jpeg;base64,{image_base64}"},
            {"text": "请详细解析这张PDF页面的内容，包括文字信息、表格、图表等所有重要信息。"}
        ]
    }
]

response = MultiModalConversation.call(
    model="qwen-vl-max",
    messages=messages
)

if response.status_code == 200:
    result = response.output.choices[0].message.content
    print("解析结果:")
    print(result)
else:
    print(f"API调用失败: {response.code} - {response.message}")

if os.path.exists(temp_image_path):
    os.remove(temp_image_path)
    print("-" * 50)
    print(f"临时文件 {temp_image_path} 已删除")
