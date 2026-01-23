import os
import pandas as pd
import jieba
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer #词频统计
from sklearn.neighbors import KNeighborsClassifier  #KNN模型
from dotenv import load_dotenv

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
#在项目根目录下创建.env文件，存入自己的阿里云API_KEY值
if not api_key:
    raise ValueError("没有找到阿里云的key值，请检查.env文件")

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key=api_key, # 账号绑定的
    # 大模型厂商的地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_ml(test_query:str)->str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]


def text_calssify_using_llm(text:str)->str:
    """
    文本分类（大语言模型），输入文本完成类别分类
    """
    completion = client.chat.completions.create(
        model="qwen-max",  # 模型的代号
        messages=[
            {"role": "system", "content": f"""
            
帮我进行文本分类：{text}
输出的类别只能从如下中进行选择：
Video-Play              
FilmTele-Play           
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
            """},  # 给大模型的命令，角色的定义

        ]
    )
    return completion.choices[0].message.content
if __name__ == '__main__':

    print("机器学习：",text_calssify_using_ml("我想去上海"))
    print("大语言模型：",text_calssify_using_llm("我想去上海"))

