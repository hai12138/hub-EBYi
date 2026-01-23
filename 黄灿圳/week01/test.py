from openai import OpenAI
import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
print(os.environ.get("OPENAI_API_KEY"))

# client = OpenAI(
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )
#
# try:
#     response = client.chat.completions.create(
#         model="qwen-plus",  # 或 qwen-turbo, qwen-max
#         messages=[{"role": "user", "content": "你好"}],
#         max_tokens=50
#     )
#     print("API连接成功！")
#     print("回复:", response.choices[0].message.content)
# except Exception as e:
#     print(f"连接失败: {e}")

stop_words = [
    '的', '了', '在', '是', '我', '有', '和', '就',
    '不', '人', '都', '一', '一个', '上', '也', '很',
    '到', '说', '要', '去', '你', '会', '着', '没有',
    '好', '自己', '这', '那', '中', '为', '与',
    '对', '但', '而', '或', '且', '之', '这', '那',
    '啊', '呀', '呢', '吧', '吗', '啦', '哇', '哦',
    '嗯', '呃', '唉', '呀', '嘛', '咧', '呗'
]

# 带停用词过滤的分词函数
def cut_with_stopwords(text: str) -> str :
    # 使用jieba分词
    words = jieba.lcut(text)
    # 过滤停用词
    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)

# 1. 数据加载
dataFrame = pd.read_table("dataset.csv", header=None, nrows=20000)
# 假设第一列是文本，第二列是标签
dataFrame.columns = ['text', 'label']
# 2. 中文分词
print("正在进行中文分词...")
dataFrame['cut_text'] = dataFrame['text'].apply(lambda x: " ".join(jieba.lcut(str(x))))
dataFrame['cut_text_wsw'] = dataFrame['text'].apply(lambda x: cut_with_stopwords(str(x)))
print(dataFrame[['cut_text', 'cut_text_wsw']])