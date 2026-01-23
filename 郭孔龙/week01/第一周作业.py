# 作业内容
import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
# print(dataset[1].value_counts())
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # sklearn对中文处理
# 特征提取
vector = CountVectorizer()  # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)  # 统计词表
input_feature = vector.transform(input_sententce.values)  #
# 模型训练
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
def text_classify_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    :param text:
    :return:
    """
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = vector.transform([text_sentence])
    return model.predict(text_feature)[0]


def text_classify_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    :param text:
    :return:
    """
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # https://bailian.console.aliyun.com/?tab=model#/api-key
        api_key="sk-fb2908176227421db2cf4f5d654bd7a1",  # 账号绑定的

        # 大模型厂商的地址
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行分本分类,{text}
            输出的类别只能从如下中选择:
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
"""}
        ]
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    print("机器学习的结果：",text_classify_ml("帮我看下明天的天气"))
    print("大语言模型结果：",text_classify_llm("帮我看下明天的天气"))
