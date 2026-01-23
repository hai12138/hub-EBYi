import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI


#特征提取
# 读取数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset[1].value_counts())
# 对文本进行分词处理
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# 创建CountVectorizer对象，用于提取文本特征
vector = CountVectorizer()
# 从input_sentence中学习词汇表
vector.fit(input_sentence.values)
# 将input_sentence转换为词频矩阵
input_sentence = vector.transform(input_sentence.values)

#模型训练
model = KNeighborsClassifier()
model.fit(input_sentence, dataset[1].values)

#生成大模型client
client = OpenAI(
    api_key="sk-b27ed66197bc46739b8a630fb1fee28XXX", #验证前先替换为可用的api_key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

#机器学习文本分类预测
def text_classify_ml_predict(text:str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

#大模型文本分类预测
def text_classify_llm_predict(text:str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f"""将文本：{text} 
    
    从以下类别中进行文本分类，然后返回给我最合适的类别:
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
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
#  机器学习文本分类
   ml_predict_result = text_classify_ml_predict("给我推荐一首好听的歌")
   print("机器学习预测文本分类结果:", ml_predict_result)
#  大模型文本分类
   llm_predict_result = text_classify_llm_predict("给我推荐一首好听的歌")
   print("大模型预测文本分类结果:", llm_predict_result)
   
   
   

