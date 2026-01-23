import pandas as pd
import jieba  # 中文分词
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier  # KNN
from openai import OpenAI


# 机器学习
dataset = pd.read_csv('dataset.csv', sep='\t', header=None, nrows=None)

# 对第一列的中文进行处理
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# print(dataset[1].value_counts())

# 对文本进行提取特征，默认是使用标点符号分词
# 将不定长文本转变为定长的文本
vector = CountVectorizer()
vector.fit(input_sentence.values)  # 统计词表
input_feature = vector.transform(input_sentence.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)


def text_classify_using_ml(text: str) -> str:
    text_sentence = ' '.join(jieba.lcut(text))
    text_feature = vector.transform([text_sentence])

    return model.predict(text_feature)[0]


# 大语言模型
client = OpenAI(
    api_key='sk-e0e141c7108a4cabaf4cced1f749ae89',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
)


def text_classify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model='qwen-flash',
        messages=[   消息= [
            {'role': 'system', 'content': 'You are a helpful assistant.'},{'role': 'system', 'content': 'You are a helpful assistant.'}，
            {'role': 'user', 'content': f'''
            帮我进行文本分类：{text}
            输出的类别只能从如下中进行选择：
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
            '''},
        ]
    )

    return completion.choices[0].message.content


def text_code() -> None:
    data = pd.read_csv('dataset.csv', sep='\t', names=['text', 'label'], nrows=None)
    print(data.head(10))
    print('数据集的样本维度：', data.shape)
    print(data['label'].value_counts())

    jieba.add_word('机器学习')
    print(jieba.lcut('我今天开始学习机器学习.'))


if __name__ == '__main__':
    # text_code()  # 测试代码
    text = '帮我找一部电影'
    print('待预测的文本：', text)
    print('KNN模型预测结果：', text_classify_using_ml(text))
    print('大语言模型预测结果：', text_classify_using_llm(text))

