import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier  # KNN模型
from fastapi import FastAPI
import openai

client = openai.OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-cb9f6485c6db428a8b1ccc0c3fee555a"
)
app = FastAPI()

# 获取数据源  数据源[0] 为excel第一列 数据源[1] 为excel第二列
dataSource = pd.read_csv("dataset.csv", sep="\t", header=None)
# 对数据源进行分词，并使用空格连接。  分词结果.values 为每行分词结果的一维矩阵（）
input_sententce = dataSource[0].apply(lambda x: " ".join(jieba.lcut(x)))
vector = CountVectorizer()
# 统计词表
vector.fit(input_sententce.values)
# 向量化转换   转化为特征向量
input_feature = vector.transform(input_sententce.values)
# print(input_feature)

# 定义模型
model = KNeighborsClassifier()
model.fit(input_feature, dataSource[1].values)


@app.get("/text-cls/ml")
def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = vector.transform([text_sentence])
    # model.predict(text_feature) 的值为数组 ['FilmTele-Play']
    return model.predict(text_feature)[0]


@app.get("/text-cls/llm")
def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}
                输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
                FilmTele-Play            
                Video-Play               
                Music-Play              
                Radio-Listen           
                Alarm-Update        
                Travel-Query        
                HomeAppliance-Control  
                Weather-Query          
                Calendar-Query """
             }
        ])
    return completion.choices[0].message.content

