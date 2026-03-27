"""
@Author  :  CAISIMIN
@Date    :  2026/3/15 18:03
"""
from pydantic import BaseModel, Field
import openai

client = openai.OpenAI(
    api_key="sk-e365d480f719416e8f4e317b7fa03ca1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

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

        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
                    }
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


class Translation(BaseModel):
    """文本翻译"""
    source_language: str=Field(description="原始语种")
    target_language: str=Field(description="目标语种")
    source_text: str=Field(description="待翻译文本")
result = ExtractionAgent(model_name="qwen-plus").call("帮我将good!翻译为中文!", Translation)
print(result)
# source_language='英文' target_language='中文' source_text='good!'
