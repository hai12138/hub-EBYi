import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 读取数据
data = pd.read_csv('dataset.csv', sep="\t", header=None, nrows=10000)
# print("分类频次统计", data[1].value_counts())

# 处理中文分词
input_words = data[0].apply(lambda x: " ".join(jieba.lcut(x)))
# print(input_words)

# 对文本进行特征提取
vector = CountVectorizer()
# 统计词表
vector.fit(input_words.values)
# 进行转换
input_feature = vector.transform(input_words.values)

# 训练模型
model = KNeighborsClassifier()
model.fit(input_feature, data[1].values)

def text_classify(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类型划分
    :param text: 输入文本
    :return: 文本类型
    """
    test_words = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_words])
    return model.predict(test_feature)[0]

if __name__ == '__main__':
    while True:
        sentence = input("请输入需要判定的语句：")
        print(text_classify(sentence))
