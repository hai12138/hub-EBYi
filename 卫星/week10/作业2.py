import base64
import fitz  # PyMuPDF
from openai import OpenAI


def pdf_first_page_to_png(pdf_path, output_image_path, zoom=2.0):
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        raise ValueError("PDF 文件为空，无法解析第一页。")

    page = doc[0]
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(output_image_path)
    doc.close()


def encode_image_to_base64(image_path):
    """
    将本地图片转为 Base64
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    pdf_path = r"D:\Application_software\Python_workspace\NLP\Week10\语言模型基础.pdf"
    output_image_path = r"D:\Application_software\Python_workspace\NLP\Week10\language_model_page1.png"
    api_key = "sk-2583d9d000d642e98254164d7aeb532d"
    pdf_first_page_to_png(pdf_path, output_image_path, zoom=2.0)
    print(f"PDF 第1页已保存为图片：{output_image_path}")
    base64_image = encode_image_to_base64(output_image_path)
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "qwenvl markdown"
                    }
                ]
            }
        ]
    )
    result = completion.choices[0].message.content
    print("\n===== Qwen-VL 解析结果 =====\n")
    print(result)


if __name__ == "__main__":
    main()
