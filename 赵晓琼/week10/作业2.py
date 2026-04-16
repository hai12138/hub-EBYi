'''
使用云端的Qwen-VL对本地的pdf第一页进行解析
'''
from openai import OpenAI
import os
from pdf2image import convert_from_path
import tempfile
import os
import io
import base64

# pdf转图片（只取第一页）
def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    return images[0]

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG") # 保存为 PNG 格式流
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

pdf_file = './流畅的Python.pdf'
image = pdf_to_image(pdf_file)

# 初始化OpenAI客户端
client = OpenAI(
    # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
    api_key = 'sk-1f98e005abc833124b42a0bf3ea03f001d521',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)2

reasoning_content = ""  # 定义完整思考过程
answer_content = ""     # 定义完整回复
is_answering = False   # 判断是否结束思考过程并开始回复

# 创建聊天完成请求
img_url = image_to_base64(image)
completion = client.chat.completions.create(
    model="qwen3-vl-plus",  # 此处以 qvq-max 为例，可按需更换模型名称
    messages=[
        {
            "role": "user",
            "content": [
                {
                   "type": "image_url",
                   "image_url": img_url
                },
                {"type": "text", "text": "这是一份文档的首页，请详细解析内容，提取所有信息"},
            ],
        },
    ],
    stream=True,
    # 解除以下注释会在最后一个chunk返回Token使用量
    # stream_options={
    #     "include_usage": True
    # }
)

print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    # 如果chunk.choices为空，则打印usage
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content

# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(reasoning_content)
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(answer_content)