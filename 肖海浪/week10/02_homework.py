import pdfplumber
import requests


# Step 1: 提取PDF第一页的内容
def extract_first_page_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        # 提取第一页的文本
        first_page = pdf.pages[0]
        text = first_page.extract_text()
    return text


# Step 2: 发送文本到 Qwen API 进行解析
def parse_with_qwen_vl(text, api_url, api_key):
    # 准备请求数据
    data = {
        "text": text,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, json=data, headers=headers)

    # 检查响应状态
    if response.status_code == 200:
        result = response.json()  # 假设返回的结果是JSON格式
        return result
    else:
        print(f"Error: {response.status_code}")
        return None


# 主函数
def main():
    pdf_path = "test.pdf"  # 本地PDF文件路径
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # Qwen API解析地址
    api_key = "sk-3fa48d768ed74074bf4ba29d428c2c76"  # 替换为能够使用的API密钥

    # 提取PDF的第一页文本
    text = extract_first_page_pdf(pdf_path)
    print("Extracted Text from PDF:")
    print(text)

    # 调用Qwen-VL解析API
    result = parse_with_qwen_vl(text, api_url, api_key)

    if result:
        print("Parsed Result from Qwen-VL:")
        print(result)


if __name__ == "__main__":
    main()