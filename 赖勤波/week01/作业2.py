"""
使用 dataset.csv数据集完成文本分类操作，需要尝试2种不同的模型。（注意：这个作业代码实操提交）
"""

import pandas as pd
import jieba
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

#利用pandas读取数据
dataset=pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
# print(dataset.head())

#使用jieba进行中文分词，单词之间用空格分开
input_sententce=dataset[0].apply(lambda x:" ".join(jieba.lcut(x)))
# print(input_sententce)

vector = CountVectorizer()
# print(vector)
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
# print(input_feature)
#加载KNN模型
model_knn = KNeighborsClassifier()
#knn模型训练
model_knn.fit(input_feature, dataset[1].values)
# print(model)

# test_query = "帮我导航到天安门"
# test_sentence = " ".join(jieba.lcut(test_query))
# test_feature = vector.transform([test_sentence])
# print("待预测的文本：", test_query)
# print("KNN模型预测的结果：",model_knn.predict(test_feature))

#svm模型
model_svm = svm.SVC()
model_svm.fit(input_feature, dataset[1].values)
# print("SVM模型预测的结果：" , model_svm.predict(test_feature))

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-533a30b9e11e4ae79076c5165f7074d6", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def txt_classify_using_ml_knn(text:str)->str:
    """
    使用机器学习中的knn算法进行文本分类，输入文本字符串，输出文本类别
    """
    text_sententce = " ".join(jieba.lcut(text))
    test_feature = vector.transform([text_sententce])
    return model_knn.predict(test_feature)

def txt_classify_using_ml_svm(text:str)->str:
    """
    使用机器学习中的svm算法进行文本分类，输入文本字符串，输出文本类别
    """
    text_sententce_svm = " ".join(jieba.lcut(text))
    test_feature_svm = vector.transform([text_sententce_svm])
    return model_svm.predict(test_feature_svm)

def txt_classify_using_llm(text:str)->str:
    """
    使用大语言模型来进行文本分类，输入文本字符串，输出文本类别
    """
    completion=client.chat.completions.create(
        model = "qwen3-max",   #模型代号
        messages = [{"role":"user", "content":f"帮我进行文本分类:{text}"}]
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    input_str = "帮我导航到北京天安门"
    input_str1 = "帮我播放音乐"
    print("machine learning KNN：", txt_classify_using_ml_knn(input_str))
    print("machine learning SVM：", txt_classify_using_ml_svm(input_str))
    print("large language model:",txt_classify_using_llm(input_str))
    print("machine learning KNN：", txt_classify_using_ml_knn(input_str1))
    print("machine learning SVM：", txt_classify_using_ml_svm(input_str1))
    print("large language model:", txt_classify_using_llm(input_str1))


