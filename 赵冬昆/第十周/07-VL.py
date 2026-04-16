# 1. 导入必要的库
import base64
import io
from openai import OpenAI
from pdf2image import convert_from_path
import os

# --- 配置区 (请替换为你的信息) ---
# 假设你已经设置了环境变量 DASHSCOPE_API_KEY
# 如果没有，请在这里写死：api_key="sk-你的key"
client = OpenAI(
    api_key="sk-0c18954972f8450c95480d5c3c0a82f6",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MODEL_NAME = "qwen2.5-vl-7b-instruct"  # 云端模型名，也可以用 qwen2.5-vl-7b-instruct


def pdf_page_to_base64(pdf_path, page_number=0):
    """
    将本地 PDF 的指定页码转换为 Base64 编码字符串
    """
    # 1. 使用 pdf2image 将 PDF 的指定页转为 PIL Image
    # 注意：poppler 需要安装并配置环境变量
    images = convert_from_path(pdf_path, dpi=150)

    if page_number >= len(images):
        raise ValueError(f"PDF 只有 {len(images)} 页，无法处理第 {page_number + 1} 页")

    page_image = images[page_number]

    # 2. 将 PIL Image 转换为 Base64 字符串
    buffer = io.BytesIO()
    page_image.save(buffer, format="JPEG")  # 转为 JPEG 格式流
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_str


def analyze_pdf_page(pdf_path, prompt="请提取并总结这一页的内容"):
    """
    主函数：解析 PDF 第一页
    """
    print("正在处理 PDF 文件...")

    # Step 1: 转换第一页 (索引 0)
    img_base64 = pdf_page_to_base64(pdf_path, page_number=0)

    # Step 2: 构造消息 (Message)
    # 注意：云端 API 通常需要 "data:image/jpeg;base64," 前缀
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    # Step 3: 调用云端 API
    print("正在调用云端 Qwen-VL 模型...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1024
    )

    # Step 4: 返回结果
    result = response.choices[0].message.content
    return result


# --- 执行作业 ---
if __name__ == "__main__":
    # 设置你的本地 PDF 路径
    LOCAL_PDF_PATH = "../data/interview_questions.pdf"  # 修改为你的实际路径

    # 定义任务指令 (Prompt)
    TASK_PROMPT = "请详细描述这张图片中的内容。如果是文档，请提取所有的文字并进行总结。"

    # 运行解析
    try:
        final_result = analyze_pdf_page(LOCAL_PDF_PATH, TASK_PROMPT)
        print("\n" + "=" * 50)
        print("云端模型解析结果：")
        print("=" * 50)
        print(final_result)
    except Exception as e:
        print(f"作业执行出错: {e}")
        print("请检查：1. PDF路径是否正确 2. 是否安装了poppler 3. API Key是否设置")