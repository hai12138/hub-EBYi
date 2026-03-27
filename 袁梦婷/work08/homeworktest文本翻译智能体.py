
from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-4f1b5ce54eb742539c58e557612243c5", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
tools = [
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "文本翻译工具，将文本从一种语言翻译为另一种语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "sourceLanguage": {
                        "description": "源语言（原始语种）",
                        "title": "SourceLanguage",
                        "type": "string",
                    },
                    "targetLanguage": {
                        "description": "目标语言",
                        "title": "TargetLanguage",
                        "type": "string",
                    },
                    "sourceText": {
                        "description": "待翻译的文本",
                        "title": "SourceText",
                        "type": "string",
                    },
                },
                "required": ["sourceLanguage", "targetLanguage", "sourceText"],
            },
        },
    }
]

# ... existing code ...
messages = [
    {
        "role": "user",
        "content": "帮我将good！翻译为中文"
    }
]
response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

tool_call = response.choices[0].message.tool_calls[0]
function_args = eval(tool_call.function.arguments)

print(f"原始语种：{function_args['sourceLanguage']}")
print(f"目标语种：{function_args['targetLanguage']}")
print(f"待翻译的文本：{function_args['sourceText']}")


def execute_translation(sourceLanguage, targetLanguage, sourceText):
    """执行实际的翻译操作"""
    translation_messages = [
        {
            "role": "user",
            "content": f"请将以下{sourceText }从{sourceLanguage}翻译为{targetLanguage}：{sourceText}"
        }
    ]

    translation_response = client.chat.completions.create(
        model="qwen-plus",
        messages=translation_messages,
    )

    return translation_response.choices[0].message.content


result = execute_translation(
    function_args['sourceLanguage'],
    function_args['targetLanguage'],
    function_args['sourceText']
)

print(f"\n翻译结果：{result}")
