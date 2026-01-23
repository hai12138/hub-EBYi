import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # logistic regression
from openai import OpenAI

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

tool1 = CountVectorizer()
input_feature = tool1.fit_transform(input_sentence)

#-------------------------逻辑回归------------------------------------------
model1 = LogisticRegression()
model1.fit(input_feature, dataset[1].values)
#-------------------------逻辑回归------------------------------------------

#-------------------------knn------------------------------------------
model2 = KNeighborsClassifier()
model2.fit(input_feature, dataset[1].values)
#-------------------------knn------------------------------------------

client = OpenAI(
    api_key="sk-8fc5397a71fe4d3bac20e6b029eb06f2",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def text_classify_logistic_regression(test:str)->str:
    test_sentence = " ".join(jieba.lcut(test))
    test_feature = tool1.transform([test_sentence])
    return model2.predict(test_feature)[0]

def text_classify_llm(test:str)->str:
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{test}

    输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
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
    return completion.choices[0].message.content


if __name__ == "__main__" :
    print("机器学习_逻辑回归: ", text_classify_logistic_regression("播放林子祥的《最爱是谁》"))
    print("机器学习_knn: ", text_classify_knn("播放林子祥的《最爱是谁》"))
    print("大模型: ", text_classify_llm("播放林子祥的《最爱是谁》"))
    print("机器学习_逻辑回归: ", text_classify_logistic_regression("帮我导航到虎门公园"))
    print("机器学习_knn: ", text_classify_knn("帮我导航到虎门公园"))
    print("大模型: ", text_classify_llm("帮我导航到虎门公园"))

    """
    结果：
    机器学习:  FilmTele-Play
    大模型:  Music-Play
    机器学习:  Travel-Query
    大模型:  Travel-Query
    """
