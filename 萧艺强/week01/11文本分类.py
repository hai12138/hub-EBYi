import pandas as pd
import jieba #中文分词
from sklearn.feature_extraction.text import CountVectorizer #词频统计
from sklearn.neighbors import KNeighborsClassifier #KNN
from openai import OpenAI
import torch

dataset = pd.read_csv('dataset.csv', encoding='utf-8', header=None
            , nrows=1000, sep='\t')
# print(dataset[1].value_counts())
input_sententce = dataset[0].apply(lambda x: ''.join(jieba.lcut(x)))#分词
vectorizer = CountVectorizer()
vectorizer.fit(input_sententce.values)#统计词频
input_feature = vectorizer.transform(input_sententce.values)#转换为特征（1000，词表大小）

knn = KNeighborsClassifier()
knn.fit(input_feature, dataset[1].values)#knn模型训练

def text_calssify_using_ml(text: str) -> str:
    '''
    机器学习预测
    '''
    s = ' '.join(jieba.lcut(text))
    feature = vectorizer.transform([s])
    return knn.predict(feature)[0]#knn模块预测


client = OpenAI(api_key='sk-5b9a83b8784a496ea31ffdf6534628eb'
                , base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
                )

def text_calssify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

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
    Calendar-Query      
    TVProgram-Play      
    Audio-Play       
    Other             
    """},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    print(torch.tensor(1))
    print(text_calssify_using_ml('帮我导航到天安'))
    print(text_calssify_using_llm('帮我导航到天安'))