import os

# pip install openai
from openai import OpenAI
import jieba
import pandas as pd

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-7b7d19ceb42a481baf3e5b6868cab18c", # 账号绑定的

    # 大模型厂商的地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-flash", # 模型的代号

    messages=[
        {"role": "system", "content": "You are a helpful assistant."}, # 提示词，用户一般看不到，给大模型的命令，角色的定义
        {"role": "user", "content": "你是谁？"},  # 用户的提问
        {"role": "user", "content": "你是谁？"},  # 用户的提问
    ]
)

def text_calssify_use_llm(text:str)->str:
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-flash",  # 模型的代号

        messages=[
            {"role": "system", "content": f"""帮我进行分类:{text} 
            输出的类别只能从以下类别中选择
            'FilmTele-Play', 'Video-Play', 'Music-Play', 'Radio-Listen', 
            'Alarm-Update', 'Travel-Query', 'HomeAppliance-Control', 'Weather-Query', 
            'Calendar-Query', 'TVProgram-Play', 'Audio-Play', 'Other'
            """},  # 提示词，用户一般看不到，给大模型的命令，角色的定义
        ]
    )
    return completion.choices[0].message.content
