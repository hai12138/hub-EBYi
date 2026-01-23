import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI
import os

# 机器学习KNN
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
# print(dataset.head(10))
# print(dataset[1].value_counts())
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.cut(x))) #用空格将分词结果连接起来
# print(dataset[1].value_counts())
# 将不定长文本转换为 维度相同的向量，总的维度和词表大小相关，用于做词频统计
vector = CountVectorizer()
vector.fit(input_sententce.values)  #分析整个文本数据集，构建词汇表（vocabulary）,学习数据的特征
input_feature = vector.transform(input_sententce.values)  # 8184*10000

model = KNeighborsClassifier()
model.fit(input_feature,dataset[1].values)


def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

# 大语言模型
def text_calssify_using_llm(text: str) -> str:
    """
        文本分类（大语言模型），输入文本完成类别划分
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)

    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": f"""帮我进行文本分类：{text}
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
        """},
                  ],stream=False
    )
    return completion.choices[0].message.content.strip()






print("机器学习: ", text_calssify_using_ml("帮我导航到万达广场"))
print("大语言模型: ", text_calssify_using_llm("帮我导航到万达广场"))
