"""
@Author  :  CAISIMIN
@Date    :  2026/3/29 21:30
"""
from io import BytesIO

from pdf2image import convert_from_path
from typing import Dict, List
import openai
import base64

client = openai.OpenAI(
    api_key="sk-e365d480f719416e8f4e317b7fa03ca1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# pdf转图片
def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Dict]:
    # 转换 PDF 为图片列表
    pages = convert_from_path(pdf_path, dpi=dpi)

    images = []
    # 并行
    for i, image in enumerate(pages):
        # Save image to bytes buffer
        buffer = BytesIO()  # 创建内存缓冲区
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)  # 指针回到开头，方便后续读取

        # Get original PDF page size in inches (approximate)
        # 根据像素和 DPI 换算物理尺寸
        width_inch = image.width / dpi
        height_inch = image.height / dpi

        images.append({
            'page_number': i + 1,
            'image': buffer,
            'content_type': 'image/jpeg',
            'width': image.width,  # 宽度（像素）
            'height': image.height,  # 高度（像素）
            'dpi': dpi,
            'width_inch': width_inch,  # 宽度（英寸）
            'height_inch': height_inch,  # 高度（英寸）
            'size_bytes': buffer.getbuffer().nbytes  # 图片字节大小
        })

    return images


def qwen_vl(image: str):
    response = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": [
                {   "type": "image_url",
                     "image_url":{
                         "url": f"data:image/jpeg;base64,{image}"
                     }
                 },
                {
                    "type": "text",
                    "text": "请详细描述这张图片的内容"
                }
            ]}
        ]
    )
    return response.choices[0].message.content


def parse_pdf_with_qwen_vl(pdf_path: str):
    print(f"正在处理pdf文档{pdf_path}")

    images = pdf_to_images(pdf_path)
    print(f"共{len(images)}页")

    # 处理第一页, 获取BytesIO对象
    first_image = images[0]['image']

    # 获取图片的二进制数据
    image_base64 = base64.b64encode(first_image.getvalue()).decode('utf-8')

    # 解析第一页
    print(qwen_vl(image_base64))


if __name__ == "__main__":
    pdf_path = "./BAT机器学习面试题库.pdf"
    parse_pdf_with_qwen_vl(pdf_path)
