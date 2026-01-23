import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI

dataset = pd.read_table("dataset.csv", sep="\t", header=None, nrows=10000)


input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 进行转换 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

def predict_text(text):
    """预测单个文本"""
    # 分词
    cut_text = " ".join(jieba.lcut(str(text)))
    text_features = vector.transform([cut_text])
    # 预测
    prediction = model.predict(text_features)
    probability = model.predict_proba(text_features)

    return prediction[0], probability[0]

# 示例：预测新文本
test_text = "播放周杰节的歌"
pred_label, pred_prob = predict_text(test_text)
print(f"文本: {test_text}")
print(f"预测标签: {pred_label}")
print(f"预测概率: {pred_prob}")