from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-533a30b9e11e4ae79076c5165f7074d6", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

"""
文本翻译智能体，能自动生成tools的json，实现信息信息抽取
"""
class TextTranslateAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # tools = [
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "TextTranslate",
        #             "description": "自动识别需要翻译的文本（原始语种、目标语种，待翻译的文本）",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "OriLanguage": {
        #                         "description": "原始语种",
        #                         "title": "OriLanguage",
        #                         "type": "string",
        #                     },
        #                     "TargetLanguage": {
        #                         "description": "目标语种",
        #                         "title": "TargetLanguage",
        #                         "type": "string",
        #                     },
        #                     "OriText": {
        #                         "description": "待翻译的文本",
        #                         "title": "OriText",
        #                         "type": "string",
        #                     },
        #                 },
        #                 "required": ["OriLanguage", "TargetLanguage", "OriText"],
        #             },
        #         },
        #     }
        # ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],  # 工具名字
                    "description": response_model.model_json_schema()['description'],  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],  # 参数说明
                        "required": response_model.model_json_schema()['required'],  # 必须要传的参数
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
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

class TextTranslate(BaseModel):
    """文本翻译智能体"""
    OriLanguage: str = Field(description='原始语种')
    TargetLanguage: str = Field(description='目标语种')
    OriText: str = Field(description='待翻译的文本')


result = TextTranslateAgent(model_name = "qwen-plus-2025-07-28").call('帮我将good！翻译为中文', TextTranslate)
print(result)   


#运行结果：OriLanguage='en' TargetLanguage='zh' OriText='good!'
