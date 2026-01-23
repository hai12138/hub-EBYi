import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier


def text_classify_use_ml(text: str) -> str:

    # 读取数据
    dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)

    # 提取文本特征
    input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # 利用jieba对中文分词处理

    vector = CountVectorizer()  # 创建向量化实例
    vector.fit(input_sententce.values)  # 创建词汇表
    input_feature = vector.transform(input_sententce.values)  # 向量化

    # 创建向量化器并构建特征
    vector = CountVectorizer() # 创建向量化实例
    vector.fit(input_sententce.values) #创建词汇表
    input_feature = vector.transform(input_sententce.values) #向量化

    # 训练KNN模型
    model = KNeighborsClassifier()
    model.fit(input_feature, dataset[1].values)

    # 对输入文本进行处理和预测
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])

    # 返回预测结果
    return model.predict(test_feature)[0]
