import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("dataset.csv",sep="\t",header=None,nrows=500)
newdata = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = TfidfVectorizer()
vector.fit(newdata.values)
X = vector.transform(newdata.values)
log_reg = LogisticRegression()
log_reg.fit(X,dataset[1].values)
print("Log_reg模型预测结果: ",log_reg.predict(vector.transform([" ".join(jieba.lcut("随便播一个"))])))