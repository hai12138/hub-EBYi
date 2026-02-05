import os

# pip install openai
from openai import OpenAI

client = OpenAI(
    # deepseek 账号的api key
    api_key="sk-d41b486adabf4888a482e2fe98abfca7",

    # 大模型厂商的地址
    base_url="https://api.deepseek.com",
)

completion = client.chat.completions.create(
    model="deepseek-chat", # 模型的代号

    messages=[
        {"role": "system", "content": """ 这是我的所有分类，一会我再次提问时，请按这个分类回复：
赶快帮我播放江苏新闻广播的电台节目高爽说法	Radio-Listen
看一眼去往南平的动车哪天有票啊	Travel-Query
快给我查询最快去一班长春去上海的机票还有几张	Travel-Query
嗯呐朱莉主演的电视剧朱莉快逃播放给我看看可以吗	FilmTele-Play
查一下武汉至成都的机票价格。	Travel-Query
建立提醒5月19号上网课	Alarm-Update
来一个战争片雪豹给我吧	FilmTele-Play
香港降雨情况怎么样	Weather-Query
        """},
        {"role": "user", "content": "今天深圳的天气怎么样"},
    ]
)

# 返回分类
print(completion.choices[0].message.content)
