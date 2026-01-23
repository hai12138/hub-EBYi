import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN

from openai import OpenAI

import os
from http import HTTPStatus
import dashscope
from dashscope import Generation
dashscope.api_key = "sk-b8f798cd51ef4b9da24d7af27324a179"  # 也可以通过环境变量设置：os.environ['DASHSCOPE_API_KEY'] = "xxx"


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-b8f798cd51ef4b9da24d7af27324a179", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


dataset = pd.read_csv("D:\BaiduNetdiskDownload\week01\Week01\dataset.csv", sep="\t", header=None, nrows=10000)

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

def text_classify_using_ml(text: str) -> str: #输入文本:字符串  输出文本->字符串
    """
    文本分类(机器学习),输入文本完成类别划分
    :param text:
    :return:
    """
    test_sentence = " ".join(jieba.lcut(text))
    print(test_sentence)
    print('----')
    test_feature = vector.transform([test_sentence])
    print(test_feature)
    return model.predict(test_feature)[0]

def text_classify_using_llm(text: str) -> str: #输入文本:字符串  输出文本->字符串
    """
    文本分类(大语言模型),输入文本完成类别划分
    :param text:
    :return:
    """
    completion = client.chat.completions.create(
        model="qwen-max",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

    输出的类别只能从如下中进行选择， 除了类别之外下列的类别，不要其他输出内容。
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




def call_qwen(text: str) -> str:
    # 调用千问大模型（model可选qwen-turbo/qwen-plus/qwen-max等）
    response = Generation.call(
        model='qwen-max',
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}
         
                输出的类别只能从如下中进行选择， 除了类别之外下列的类别，不要其他输出内容。
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
            ],
        result_format='message',  # 返回格式和OpenAI兼容
        stream=False,  # 非流式输出
        temperature=0.7,  # 生成随机性
    )
    # return response.output.choices[0].message.content

    # 处理响应
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content
    else:
        re = f'调用失败：{response.code} - {response.message}'
        return re

if __name__ == '__main__':

    print('机器学习', text_classify_using_ml('帮我导航到扬州'))
    print('openai调用大语言模型', text_classify_using_llm('帮我导航到扬州'))
    print('dashscope调用大语言模型', call_qwen('帮我导航到扬州'))