'''
参考 04_Pydantic与Tools.py 中的tools的实现，
搭建出一个文本翻译智能体，自动识别需要翻译的文本（原始语种、目标语种，待翻译的文本）。
帮我将good！翻译为中文 -》 原始语种、目标语种，待翻译的文本
'''
from pydantic import BaseModel, Field
import openai
import json
client = openai.OpenAI(
    # base_url='http://localhost:11434/v1/',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-5b9a83b8784a496ea31ffdf653462eb"
)
class TranslateAgent:
    def __init__(self, model_name: str='qwen2.5:7b'):
        self.model_name = model_name
    def translate(self, text: str, response_model) -> str:
        messages = [
            {
                'role': 'user',
                'content': text,
            }
        ]
        tools = [
            {
                "type":"function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
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
            tool_choice='auto'
        )
        print(response.choices)
        if (response.choices[0].message.tool_calls
                and response.choices[0].message.tool_calls[0].function):
            function = response.choices[0].message.tool_calls[0].function
            arguments = function.arguments
            arguments = json.loads(arguments)
            name = function.name
            print(arguments,name)
            if name in function_map:
                local_func = function_map[name]
                result = local_func(**arguments)
                return result
        return None



class TranslateModel(BaseModel):
    '''翻译文本提取模型'''
    source_language: str = Field(description='原始语种')
    target_language: str = Field(description='目标语种')
    text_to_translate: str = Field(description='待翻译的文本')

from agents import Agent,Runner,set_tracing_disabled,set_default_openai_api
set_default_openai_api('chat_completions')
set_tracing_disabled(True)

agent = Agent(model='qwen2.5:7b',name='translated',instructions='你是翻译专家')

def translate(source_language: str, target_language: str,
              text_to_translate: str):
    input = f'''原始语种是{source_language}，目标语种是{target_language}
    需要翻译的内容是：{text_to_translate}'''
    translated_text = Runner.run_sync(agent, input)
    map = {
        'source_language': source_language,
        'target_language': target_language,
        'text_to_translate': text_to_translate,
        'translated_text': translated_text.final_output
    }
    return map
function_map = {}
function_map['TranslateModel'] = translate

text = '''请将以下的文本翻译成德文
文本内容是：小强是小王的好朋友'''
result = TranslateAgent("qwen-max").translate(text,TranslateModel)
print(result)