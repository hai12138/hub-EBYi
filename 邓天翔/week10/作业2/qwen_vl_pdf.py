from openai import OpenAI
import fitz  # PyMuPDF
import os

# 设置API密钥
api_key = "sk-37e05617b8574cd2ad80371a106c0d81"  # 请替换为您的实际API密钥

# 初始化OpenAI客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 打开PDF并提取第一页为图片
pdf_path = "八斗学院精品班@NLP与大模型+课程笔记-260118.pdf"
doc = fitz.open(pdf_path)
page = doc[0]  # 第一页（索引从0开始）

# 渲染页面为图片
mat = fitz.Matrix(2.0, 2.0)  # 设置缩放比例，提高清晰度
pix = page.get_pixmap(matrix=mat)

# 将图片转换为base64
import base64
img_byte_arr = pix.tobytes(output='JPEG')
base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

# 关闭文档释放资源
doc.close()

reasoning_content = ""  # 定义完整思考过程
answer_content = ""     # 定义完整回复
is_answering = False   # 判断是否结束思考过程并开始回复

# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen3-vl-235b-a22b-thinking",  # 使用 qwen3-vl-235b-a22b-thinking 模型
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
                {"type": "text", "text": "请解析这张PDF页面的内容，并总结关键信息。"},
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

# 不再需要临时文件

# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(reasoning_content)
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(answer_content)