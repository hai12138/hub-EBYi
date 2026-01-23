import pandas as pd
import numpy as np
import jieba
import random  # 新增：随机数库，用于随机抽取样本
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI


# 模型1：机器学习模型
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print("数据集各类别数量分布：")
print(dataset[1].value_counts())

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 进行转换 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

# 模型2：qwen-flash
client = OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    # 替换为自己的api_key
    api_key="sk-411bcb4cd2a940f69a02a6d239b9ec4e", # 账号绑定，用来计费的
    # 大模型厂商的地址，阿里云，不用改动
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_llm(text: str) -> str:
    """
    LLM,实现文本分类
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择，除了类别之外不要输出任何多余内容，请给出最合适的类别。
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
    return completion.choices[0].message.content.strip() # 去除首尾空格，防止返回值带空格影响结果


def random_predict_from_dataset():
    """
    随机从数据集抽取1条样本，使用两个模型进行推理，并打印结果对比
    """
    # 1. 随机生成一个行索引，范围：0 ~ 数据集最后一行的索引
    random_idx = random.randint(0, len(dataset)-1)
    # 2. 根据随机索引读取对应的 文本内容 和 真实标签
    random_text = dataset.iloc[random_idx, 0]  # iloc[行号,列号] 0列=第一列=文本内容
    true_label = dataset.iloc[random_idx, 1]   # 1列=第2列=真实分类标签
    # 3. 调用两个模型分别做预测
    ml_pred_label = text_calssify_using_ml(random_text)  # KNN模型预测
    llm_pred_label = text_calssify_using_llm(random_text) # 千问大模型预测

    print("随机抽取的样本索引", random_idx)
    print("待分类文本内容：", random_text)
    print("文本真实标签：", true_label)
    print("KNN机器学习模型预测标签：", ml_pred_label)
    print("Qwen-Flash大模型预测标签：", llm_pred_label)

if __name__ == '__main__':
    random_predict_from_dataset()
