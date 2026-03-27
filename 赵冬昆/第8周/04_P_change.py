from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-0c18954972f8450c95480d5c3c0a82f6", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 本质是function call
# 可以传入多个待选函数，让大模型选择其中一个
# 传的是我们的函数的描述，让大模型选择，生成调用这个函数的传入参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "Ticket",
            "description": "根据用户提供的信息查询火车时刻",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "description": "要查询的火车日期",
                        "title": "Date",
                        "type": "string",
                    },
                    "departure": {
                        "description": "出发城市或车站",
                        "title": "Departure",
                        "type": "string",
                    },
                    "destination": {
                        "description": "要查询的火车日期",
                        "title": "Destination",
                        "type": "string",
                    },
                },
                "required": ["date", "departure", "destination"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "你能帮我查一下2024年1月1日从北京南站到上海的火车票吗？"
    }
]

# 大模型选择了一个函数，生成了函数的调用过程， 这也是agent 的核心功能
response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools, # 生成函数的调用方式，并不是所有的模型都支持（某些比较小的模型不支持）
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls[0].function)


"""
这个智能体（不是满足agent所有的功能），能自动生成tools的json，实现信息信息抽取
指定写的tool的格式
"""
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 传入需要提取的内容，自己写了一个tool格式
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class Text(BaseModel):
    """文本问答内容解析"""
    search: bool = Field(description="是否需要搜索")
    keywords: List[str] = Field(description="待选关键词")
    intent: Literal['music', 'app', 'weather'] = Field(description="意图")
result = ExtractionAgent(model_name = "qwen-plus").call('汽车发动和轮胎出故障了，如何处理？', Text)
print(result)

class Ticket(BaseModel):
    """根据用户提供的信息查询火车时刻"""
    date: str = Field(description="要查询的火车日期")
    departure: str = Field(description="出发城市或车站")
    destination: str = Field(description="要查询的火车日期")
result = ExtractionAgent(model_name = "qwen-plus").call("你能帮我查一下2024年1月1日从北京南站到上海的火车票吗？", Ticket)
print(result)



class Text(BaseModel):
    """抽取句子中的的单词，进行文本分词"""
    keyword: List[str] = Field(description="单词")
result = ExtractionAgent(model_name = "qwen-plus").call('小强是小王的好朋友。谢大脚是长贵的老公。', Text)
print(result)


class Text(BaseModel):
    """分析文本的情感"""
    sentiment: Literal["正向", "反向"] = Field(description="情感类型")
result = ExtractionAgent(model_name = "qwen-plus").call('我今天很开心。', Text)
print(result)


class Text(BaseModel):
    """分析文本的情感"""
    sentiment: Literal["postivate", "negative"] = Field(description="情感类型")
result = ExtractionAgent(model_name = "qwen-plus").call('我今天很开心。', Text)
print(result)


class Text(BaseModel):
    """抽取实体"""
    person: List[str] = Field(description="人名")
    location: List[str] = Field(description="地名")
result = ExtractionAgent(model_name = "qwen-plus").call('今天我和徐也也去海淀吃饭，强哥也去了。', Text)
print(result)


class Text(BaseModel):
    """抽取句子中所有实体之间的关系"""
    source_person: List[str] = Field(description="原始实体")
    target_person: List[str] = Field(description="目标实体")
    relationship: List[Literal["朋友", "亲人", "同事"]] = Field(description="待选关系")
result = ExtractionAgent(model_name = "qwen-plus").call('小强是小王的好朋友。谢大脚是长贵的老公。', Text)
print(result)


class Text(BaseModel):
    """抽取句子的摘要"""
    abstract: str = Field(description="摘要结果")
result = ExtractionAgent(model_name = "qwen-plus").call("20年来，中国探月工程从无到有、从小到大、从弱到强。党的十八大后，一个个探月工程任务连续成功，不断刷新世界月球探测史的中国纪录嫦娥三号实现我国探测器首次地外天体软着陆和巡视探测，总书记肯定“在人类攀登科技高峰征程中刷新了中国高度”；", Text)
print(result)


class Text(BaseModel):
    """文本问答内容解析"""
    search: bool = Field(description="是否需要搜索")
    keywords: List[str] = Field(description="待选关键词")
    intent: Literal["查询客服问题", "查询产品问题", "查询系统问题", "其他"] = Field(description="意图")
result = ExtractionAgent(model_name = "qwen-plus").call('汽车发动和轮胎出故障了，如何处理？', Text)
print(result)


class Text(BaseModel):
    """文本问答内容解析"""
    time: List[str] = Field(description="时间")
    particate: List[str] = Field(description="选手")
    competition: List[str] = Field(description="赛事名称")
result = ExtractionAgent(model_name = "qwen-plus").call('2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办。', Text)
print(result)

class Translate(BaseModel):
    """翻译"""
    text: str =  Field(description="需要翻译的文本")
    source_lang: Literal["en", "zh"] = Field(description="原始语种，例如：en（英语）、zh（中文）")
    target_lang: Literal["en", "zh"] = Field(description="目标语种，例如：zh（中文）、en（英语）")
result = ExtractionAgent(model_name = "qwen-plus").call('我今天很good。', Translate)
print(result)

