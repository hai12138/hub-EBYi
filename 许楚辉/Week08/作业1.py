from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-8fc5397a71fe4d3bac20e6b029eb06f2", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


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

class Translate(BaseModel):
    """自动识别需要翻译的文本"""
    types: List[str] = Field(description="原始语种")
    target_type:str = Field(description="目标语种")
    intent: List[str] = Field(description="待翻译的内容")
# result = ExtractionAgent(model_name = "qwen-plus").call('我听不懂how are you和i\'am file,帮我翻译一下', Translate)
# print(result)

# 创建翻译Agent
class TranslateAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.extraction_agent = ExtractionAgent(model_name)

    def call(self, user_prompt) :
        extract_result = self.extraction_agent.call(user_prompt, Translate)
        if extract_result is None:
            return "无法识别要翻译的文本"
        
        # 翻译
        translate_messages = [
            {
                
                "role":"user",
                "content": f"你是一个专业的翻译，需要根据用户提供的文本，将其翻译为指定的目标语种。请将以下{extract_result.types}翻译为{extract_result.target_type}，\n"
                + "\n".join(extract_result.intent)
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=translate_messages,
        )
        return {
            "original_texts": extract_result.intent,
            "source_language": extract_result.types,
            "target_language": extract_result.target_type,
            "translation": response.choices[0].message.content
        }

agent = TranslateAgent("qwen-plus")
result = agent.call("我听不懂お元気ですか,帮我翻译一下")
print(result)
        
