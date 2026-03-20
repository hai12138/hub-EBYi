import os
import json
from typing import Optional

from pydantic import BaseModel, Field
from openai import OpenAI


client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)


class TranslateModel(BaseModel):
    """翻译请求提取模型"""
    source_language: str = Field(description="原始语种，如英文、中文、日文等")
    target_language: str = Field(description="目标语种，如中文、英文、德文等")
    text_to_translate: str = Field(description="待翻译的文本内容")


class TranslateAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.extract_model = model_name
        self.translate_model = model_name

    def extract_translation_request(self, user_text: str) -> Optional[dict]:
        schema = TranslateModel.model_json_schema()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_translation_request",
                    "description": "从用户输入中提取翻译任务所需参数，包括原始语种、目标语种、待翻译文本",
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema.get("required", [])
                    }
                }
            }
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个翻译任务解析助手。"
                    "你的任务是从用户输入中提取翻译请求的三个字段："
                    "source_language、target_language、text_to_translate。"
                    "如果原始语种没有明确给出，请根据待翻译文本自动判断。"
                )
            },
            {
                "role": "user",
                "content": user_text
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.extract_model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            choice = response.choices[0].message
            if not choice.tool_calls:
                return None

            function = choice.tool_calls[0].function
            if function.name != "extract_translation_request":
                return None

            arguments = json.loads(function.arguments)
            data = TranslateModel(**arguments)
            return data.model_dump()

        except Exception as e:
            print(f"参数提取失败: {e}")
            return None

    def do_translate(self, source_language: str, target_language: str, text_to_translate: str) -> Optional[str]:
        messages = [
            {
                "role": "system",
                "content": "你是专业翻译助手。请严格按照目标语种翻译，直接返回译文，不要添加解释。"
            },
            {
                "role": "user",
                "content": (
                    f"请将以下{source_language}内容翻译为{target_language}：\n"
                    f"{text_to_translate}"
                )
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.translate_model,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"翻译失败: {e}")
            return None

    def run(self, user_text: str) -> Optional[dict]:
        extracted = self.extract_translation_request(user_text)
        if not extracted:
            return None

        translated_text = self.do_translate(
            source_language=extracted["source_language"],
            target_language=extracted["target_language"],
            text_to_translate=extracted["text_to_translate"]
        )

        result = {
            "source_language": extracted["source_language"],
            "target_language": extracted["target_language"],
            "text_to_translate": extracted["text_to_translate"],
            "translated_text": translated_text
        }
        return result


if __name__ == "__main__":
    agent = TranslateAgent()

    text1 = "请将以下的文本翻译成德文，文本内容是：小强是小王的好朋友"
    result1 = agent.run(text1)
    print(result1)

    text2 = "帮我将good！翻译为中文"
    result2 = agent.run(text2)
    print(result2)
    