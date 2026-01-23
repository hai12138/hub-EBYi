import jieba  # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.naive_bayes import MultinomialNB
import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier  # KNN
from openai import OpenAI


# 使用传统模型
def run_ml(
    csv_path: str = "dataset.csv",
    nrows: int = 100,
):
    # 读数据
    dataset = pd.read_csv(csv_path, sep="\t", header=None, nrows=nrows)
    # print(dataset.head(5))

    # 分词（sklearn 的向量器默认按空格分词）
    input_sentence = dataset[0].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))

    # 向量化
    vector = CountVectorizer()
    input_feature = vector.fit_transform(input_sentence.values)

    # 训练模型替换为 MultinomialNB
    model = MultinomialNB()
    model.fit(input_feature, dataset[1].values)
    print(model)

    # 3条测试文本
    test_queries = [
        "帮我查一下明天从北京到伦敦的联航机票",
        "随机播放一首周杰伦不喜欢听的歌",
        "我上次生日所在那天是什么节日",
    ]

    print("\n预测结果:")
    for q in test_queries:
        test_sentence = " ".join(jieba.lcut(q))
        test_feature = vector.transform([test_sentence])
        pred = model.predict(test_feature)[0]
        print(f"待预测文本: {q}")
        print(f"预测结果: {pred}\n")

    return model, vector


# 使用大模型
def run_llm(
    model_name: str = "gpt-5-nano",
):
    # OpenAI 客户端
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def text_calssify_using_llm(text: str) -> str:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"""帮我进行文本分类：{text}

要求：只输出一个类别名称，不要输出任何解释或多余文字。
输出类别只能从如下中选择：
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
""",
                }
            ],
        )
        return (completion.choices[0].message.content or "").strip()

    # 示例演示
    demo_texts = [
        "帮我查一下明天从北京到伦敦的联航机票",
        "随机播放一首周杰伦不喜欢听的歌",
        "我上次生日所在那天是什么节日",
    ]
    for t in demo_texts:
        print("文本:", t)
        print("大语言模型:", text_calssify_using_llm(t))
        print("-" * 30)

    return text_calssify_using_llm


