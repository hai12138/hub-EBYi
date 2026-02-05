# hzs 2026-01-15, 作业，使用sklearn进行文本分类
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 加载文本集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=50)
print(dataset.head(2))  # 打印前5行文本

# 1、提取文本特征
# 2、构建KNN，学习提取特征和标签关系
# 3、文本预测，核对结果

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # 中文处理（友好）

verctor = CountVectorizer()  # 创建特征向量, 进行文本特征提取，默认分号符号
verctor.fit(input_sententce.values)
input_feature = verctor.transform(input_sententce.values)
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
print(model)


test_query = "帮我查下今天深圳市南山区的天气情况"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = verctor.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))
print("KNN模型预测概率: ", model.predict_proba(test_feature))

