"""
@Author  :  CAISIMIN
@Date    :  2026/1/14 21:20
"""

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 读取数据
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None, names=['text', 'label'])
print(dataset.head(5))
print("-" * 100)
# 样本量 24200
print(dataset.size)
print("-" * 100)

# jieba分词
# lcut返回list（适合小样本，可索引、切片）; cut返回生成器（内存效率高，适合处理大文件）
input_sentence = dataset['text'].apply(lambda x: " ".join(jieba.lcut(x)))
print(input_sentence[:5])
print("-" * 100)

# 划分数据集，train:test = 8:2
labels = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(
    input_sentence, labels, random_state=32, shuffle=True, test_size=0.2
)

# 向量化 TfidfVectorizer相比CountVectorizer，可降低高频无意义词的权重
vector = TfidfVectorizer()
vector.fit(X_train)
X_train_feature = vector.transform(X_train)
X_test_feature = vector.transform(X_test)

# 训练LR模型
model = LogisticRegression()
model.fit(X_train_feature, y_train)

# 预测与评估
y_pred = model.predict(X_test_feature)
acc = accuracy_score(y_test, y_pred)
print(f"准确率：{acc}")
print("-" * 100)

test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("-" * 100)
print("LR模型预测结果: ", model.predict(test_feature))
