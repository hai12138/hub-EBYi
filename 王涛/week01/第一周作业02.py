import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

data = pd.read_csv("dataset.csv", sep="\t", names=["text","label"], nrows=10000)
input_sentence = data['text'].apply(lambda x: " ".join(jieba.lcut(x)))
print(data['label'].value_counts())

vector = CountVectorizer() #对文本进行特征提取，默认是使用标点符号分词
vector.fit(input_sentence.values) #统计词频
input_feature = vector.transform(input_sentence.values)

model = KNeighborsClassifier()
model.fit(input_feature, data['label'].values) #使用提取的文本特征、数据对应的标签

client = OpenAI(api_key="sk-90dbaad7468e4c698bbb8c19e4f7b0dd",
                base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')

def text_classify_using_ml(text:str ) -> str:
    """
    文本分类,机器学习
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]



def text_classify_using_llm(text:str ) -> str:
    """
    文本分类，大语言模型
    """
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "system", "content":f"""请帮我进行文本分类{text},只输出类别，不要输出其他内容。
类别列表：
Phone-Call
Phone-Query
Phone-Control
Phone-Listen
Phone-Dial
Phone-HangUp
Phone-Mute
Phone-Unmute
Phone-VolumeUp
Phone-VolumeDown
Phone-Answer
Phone-Reject
Phone-Hold
Phone-Unhold
Phone-Transfer
Phone-SendMessage
Phone-ReadMessage
Phone-DeleteMessage
Phone-SendEmail
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
Other"""},  #提问
        ]
    )
    return  completion.choices[0].message.content


if __name__ == "__main__":
    print("机器学习:",text_classify_using_ml("帮我导航到天安门"))
    print("大语言模型:",text_classify_using_llm("帮我导航到天安门"))
