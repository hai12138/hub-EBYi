import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("dataset.csv",sep="\t",header=None,nrows=500)
newdata = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = CountVectorizer()
vector.fit(newdata.values)
X = vector.transform(newdata.values)
knn = KNeighborsClassifier()
knn.fit(X,dataset[1].values)
print("KNN模型预测结果: ",knn.predict(vector.transform([" ".join(jieba.lcut("开心时候听的歌曲"))])))