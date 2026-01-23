"""
@Author  :  CAISIMIN
@Date    :  2026/1/14 22:19
"""
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # SVC - 分类；SVR - 回归

# 读取数据
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None, names=["text", "label"])
# jieba分词
input_sentence = dataset["text"].apply(lambda x: " ".join(jieba.lcut(x)))
print(input_sentence[:5])
print("-" * 100)

# 划分数据集
labels = dataset["label"]
X_train, X_test, y_train, y_test = train_test_split(
    input_sentence, labels, test_size=0.2, shuffle=True, random_state=32
)

# 向量化
vector = TfidfVectorizer()
vector.fit(X_train)
X_train_feature = vector.transform(X_train)
X_test_feature = vector.transform(X_test)

# 训练SVM模型
model = SVC(kernel="linear") # 准确率：0.8954545454545455
# model = SVC() # 准确率：0.890495867768595
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
print("SVM模型预测结果: ", model.predict(test_feature))
