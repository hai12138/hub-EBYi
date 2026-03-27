'''
参考 04_Pydantic与Tools.py 中的tools的实现，搭建出一个文本翻译智能体，自动识别需要翻译的文本（原始语种、目标语种，待翻译的文本）。
帮我将good！翻译为中文 -》 原始语种、目标语种，待翻译的文本
'''
from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class TranslationSchema(BaseModel):
    original: str = Field(..., description="原始语种（如 'en' 或 '英文'）")
    goal: str = Field(..., description="目标语种（如 'zh' 或 '中文'）")
    text: str = Field(..., description="待翻译的文本")
    result: str = Field(..., description="译文")

tools = [{
    "type": "function",
    "function": {
        "name": "Translation",
        'description': "确定原始语种、目标语种、待翻译文本以及翻译结果",
        "parameters": {
            "type": "object",
            "properties": {
                "original": {
                    'description': '原始语种',
                    'title': "Original",
                    "type": "string"
                },
                "goal": {
                    'description': '目标语种',
                    'title': "Goal",
                    "type": "string"
                },
                "text": {
                    'description': '待翻译文本',
                    'title': "Text",
                    "type": "string"
                },
                'result': {
                    'description': '译文',
                    'title': "Result",
                    "type": "string"
                }
            },
            "required": ['original', 'goal', 'text', 'result']
        }
    }
}]

def translate_text(original, goal, text, result):
    return f"原始语种：{original}, 目标语种：{goal}, 待翻译的文本：{text}, 译文：{result}"

def run_translation_agent(user_prompt):
    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools, # 生成函数的调用方式，并不是所有的模型都支持（某些比较小的模型不支持）
        tool_choice="auto",
    )
    # 检查是否有tool_call
    choice_msg = response.choices[0].message
    if not getattr(choice_msg, 'tool_calls', None):
        # 模型没有发起工具调用，直接返回它的自然语言回复
        return {"final": choice_msg.content}
    tool_calls = choice_msg.tool_calls
    tool_response = []
    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        parsed = TranslationSchema(**args)
        print(parsed, '查看parsed的值是什么---===') # original='英语' goal='中文' text='good' result='好'
        translated = translate_text(parsed.original, parsed.goal, parsed.text, parsed.result)
        tool_response.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": translated
        })
    messages.append(choice_msg)
    messages.extend(tool_response)
    # 再次调用模型以获得最终回复
    final = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools
    )
    return {"final": final.choices[0].message.content, "tool_responses": tool_response}

# 简单运行示例
if __name__ == "__main__":
    out = run_translation_agent('请把 "good" 翻译成中文')
    print(out)
