import pymupdf, fitz
from openai import OpenAI
import base64
from PIL import Image

# 1. 初始化客户端，指向本地vLLM服务
client = OpenAI(
    api_key="sk-",  # vLLM服务通常不需要API Key，但需要提供非空值
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 您的vLLM服务地址
)

def pdf_page_to_image_pymupdf(pdf_path, page_num = 1, zoom=2.0):
    """
    将 PDF 指定页面转换为 PIL Image 对象（使用 PyMuPDF）

    参数:
        pdf_path: PDF 文件路径
        page_num: 页码（从 1 开始）
        zoom: 缩放倍数，控制输出图像分辨率

    返回:
        PIL Image 对象
    """
    doc = fitz.open(pdf_path)
    if page_num < 1 or page_num > len(doc):
        raise ValueError(f"页码超出范围：1-{len(doc)}")

    page = doc[page_num - 1]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img
def image_to_base64_str(image):
    """
    将 PIL Image 对象转换为 base64 编码的字符串。
    """
    import io
    # 将图片保存到内存缓冲区
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    # 获取字节并进行 base64 编码
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def pdf_to_image1(pdf_path, page_num=1):
    """
    将 PDF 的指定页面转换为 PIL Image 对象。
    默认转换第一页 (page_num=1)。
    """
    # 打开 PDF 文档
    doc = fitz.open(pdf_path)
    # 加载指定页面 (页码从1开始，fitz从0开始，所以减1)
    page = doc[page_num - 1]
    image_list = page.get_images(full=True)
    print(len(image_list))
    image_base64_list = []
    text = page.get_text()
    for img_index, img in enumerate(image_list):
        xref = img[0]
        print(img_index)
        print(xref)
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_base64_list.append(image_base64)
        # Image.open(io.BytesIO(image_bytes)).show()
    doc.close()
    return text,image_base64_list

def chat():
    text, image_base64_list = pdf_to_image1('../data/2307.06435v10.pdf', page_num=4)
    prompt = "Read all the text in the image."
    content = []
    page_text = [text]
    for image_base64 in image_base64_list:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    chatbot(messages,page_text)
# print('messages', messages)
# 3. 构建消息并调用
def chatbot(messages,page_text = []):
    try:
        response = client.chat.completions.create(
            model="qwen2.5-vl-3b-instruct",
            messages= messages,
            max_tokens=1024,
            temperature=0.1
        )

        # 4. 输出结果
        print("模型回复：")
        # print(response.choices)
        print(response.choices[0].message.content)
        page_text.append(response.choices[0].message.content)
        print('page_text', page_text)
    except Exception as e:
        print(f"调用模型时出错: {e}")

def chat1():
    prompt = "Read all the text in the image."
    img = pdf_page_to_image_pymupdf('../data/2307.06435v10.pdf', page_num=4)
    img.show()
    image_base64 = image_to_base64_str(img)
    content = []
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
    content.append({"type": "text", "text": prompt})
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    chatbot(messages)

if __name__ == '__main__':
    chat()
    print("=" * 30)
    chat1()
