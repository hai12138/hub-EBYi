# 2. 使用 dataset.csv 数据集完成文本分类操作，需要尝试 2 种不同的模型。

import jieba
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# 1) 读取数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
dataset.columns = ["text", "label"]
print("数据量:", len(dataset))
print(dataset.head(5))

# 2) 中文分词：将句子切词并用空格连接
seg_text = dataset["text"].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))

# 3) 提取词频特征
vector = CountVectorizer()
X = vector.fit_transform(seg_text.values)
y = dataset["label"].values

# 4) 训练两种模型
models = {
    "MultinomialNB": MultinomialNB(),
    "LinearSVC": LinearSVC()
}

for name, clf in models.items():
    clf.fit(X, y)
    print(f"{name} trained.")

# 5) 预测
test_query = "导航到最近的加油站，避开高速"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

print("\n待预测的文本:", test_query)
for name, clf in models.items():
    pred = clf.predict(test_feature)[0]
    print(f"{name} 模型预测结果: {pred}")
