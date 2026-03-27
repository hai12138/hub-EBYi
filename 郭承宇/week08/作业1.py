from pydantic import BaseModel,Field
from typing import List
from typing_extensions import Literal
import json

import openai

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class Translation(BaseModel):
    """根据原始语种将用户提供的待翻译的文本翻译为目标语种"""
    source_language:Literal["中文", "英文", "日文", "韩文", "法文", "西班牙文", "葡萄牙文", "俄文", "意大利文", "德文", "阿拉伯文", "俄文", "丹麦文", "希腊文", "荷兰文", "波兰文", "瑞典文", "泰文", "越南文", "中文繁体", "中文简体", "中文香港", "中文澳门", "中文台湾", "中文香港", "中文澳门", "中文台湾"] = Field(description="原始语种")
    target_language:Literal["中文", "英文", "日文", "韩文", "法文", "西班牙文", "葡萄牙文", "俄文", "意大利文", "德文", "阿拉伯文", "俄文", "丹麦文", "希腊文", "荷兰文", "波兰文", "瑞典文", "泰文", "越南文", "中文繁体", "中文简体", "中文香港", "中文澳门", "中文台湾", "中文香港", "中文澳门", "中文台湾"] = Field(description="目标语种")
    text:str = Field(description="待翻译的文本")



class ExtractionAgent:
    def __init__(self,model_name:str):
        self.model_name = model_name

    def call(self,user_prompt,response_model):

        # print(response_model.model_json_schema())

        messages = [
            {
                'role':'user',
                'content':f'{user_prompt},顺便将待翻译的文本抽取出来'
            }
        ]

        tools = [
            {
                'type':'function',
                'function':
                    {
                        'name':response_model.model_json_schema()['title'],
                        'description':response_model.model_json_schema()['description'],
                        'parameters':
                            {
                                'type':'object',
                                'properties':response_model.model_json_schema()['properties'],
                                'required':response_model.model_json_schema()['required'],
                            }
                    }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice='auto',
        )

        try:
            # print(response.choices)
            # print(response.choices[0].message)
            # print(response.choices[0].message.tool_calls[0].function)
            function_name = response.choices[0].message.tool_calls[0].function.name
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return function_name,response_model.model_validate_json(arguments)
        except:
            print('ERROR',response.choices[0].message)
            return None

def translate(source,target,text):
    response = client.chat.completions.create(
        model='qwen-plus',
        messages=[
            {
                'role':'user',
                'content':f'''
                你是一个翻译专家，你需要根据原始语种将用户提供的待翻译的文本翻译为目标语种。
                - 原始语种：{source}
                - 目标语种：{target}
                - 待翻译的文本：{text}
                
                并返回如下json格式的结果，
                {{
                "original":"待翻译的原始语种文本"
                "translation": "翻译结果"
                }}
                
                例子：
                {{
                "original":"I am a student.",
                "translation": "我是一个学生。"
                }}
                
                '''
            }
        ]
    )
    result = response.choices[0].message.content
    # print(result)
    result_json = json.loads(result)
    # print(result_json)

    return result_json


functions = {
    'Translation':translate
}


agent = ExtractionAgent(model_name='qwen-plus')
function_name,result = agent.call('请帮我翻译：今天天气真好，我们一起去公园吧。',Translation)
# print(result,type(result))

for name in functions:
    if name == function_name:
        translate_result = functions[name](result.source_language,result.target_language,result.text)
        print(translate_result)


