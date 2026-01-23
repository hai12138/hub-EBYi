import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer # 使用TF-IDF统计
from openai import OpenAI
from typing import Union

from fastapi import FastAPI
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
# print(dataset[1].value_counts())


input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
# print(input_sententce[:10])

vector = TfidfVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 向量化
# print(input_feature[:10]) # input_feature是一个稀疏矩阵， 稀疏矩阵的存储空间小， 节省内存

# 使用决策树
model = DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)


client = OpenAI(
    # 调用星火大模型api
    api_key="ukNypLGtaRzaqAqVZjBe:uFkfVkLjqvFuNzXeFLBN", # 账号绑定，用来计费的

    base_url="https://spark-api-open.xf-yun.com/v2",
)

@app.get("/text-cls/ml")
def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return "机器学习分类：" + model.predict(test_feature)[0]

@app.get("/text-cls/llm")
def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(

        model="spark-x",  # 模型的代号
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择， 不要输出类别以外的文字，请给出最合适的类别。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control  
Weather-Query          
Calendar-Query      
TVProgram-Play      
Audio-Play       
Other             
"""},  # 用户的提问
        ]
    )
    return "大模型分类：" + completion.choices[0].message.content

if __name__ == "__main__":
    print("机器学习: ", text_calssify_using_ml("今天天气一般"))
    print("大语言模型: ", text_calssify_using_llm("今天天气一般"))
