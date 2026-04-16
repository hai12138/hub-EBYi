from openai import OpenAI
import base64
import fitz  # PyMuPDF
import os


# --- 1. 定义 PDF 转图片的辅助函数 ---
def get_pdf_page_base64(pdf_path, page_index):
    """
    将 PDF 的特定页转换为 Base64 编码的图片字符串
    page_index 从 0 开始计算
    """
    # 打开 PDF
    doc = fitz.open(pdf_path)
    # 选择页面
    page = doc.load_page(page_index)

    # 将页面渲染为图片 (matrix 可以调整清晰度，2表示放大2倍，默认DPI较低)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

    # 将图片数据转换为字节流
    img_bytes = pix.tobytes("jpg")

    # 编码为 base64
    base64_str = base64.b64encode(img_bytes).decode('utf-8')

    doc.close()
    return base64_str
# 初始化OpenAI客户端
client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key="sk-8fc5397a71fe4d3bac20e6b029eb06f2",
    # 以下是北京地域base_url，若使用弗吉尼亚地域模型，需要将base_url换成https://dashscope-us.aliyuncs.com/compatible-mode/v1
    # 如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

reasoning_content = ""  # 定义完整思考过程
answer_content = ""     # 定义完整回复
is_answering = False   # 判断是否结束思考过程并开始回复
enable_thinking = True

local_pdf_path = "../Week10-多模态大模型.pdf"
image_base64 = get_pdf_page_base64(local_pdf_path, page_index=3)

# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
                {"type": "text", "text": "这一页讲了什么？"},
            ],
        },
    ],
    stream=True,
    # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
    # qwen3.5-plus、qwen3-vl-plus、qwen3-vl-flash可通过enable_thinking开启或关闭思考（其中qwen3.5-plus默认开启）、对于qwen3-vl-235b-a22b-thinking等带thinking后缀的模型，enable_thinking仅支持设置为开启，对其他Qwen-VL模型均不适用
    extra_body={
        'enable_thinking': enable_thinking},

    # 解除以下注释会在最后一个chunk返回Token使用量
    stream_options={
        "include_usage": True
    }
)

if enable_thinking:
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