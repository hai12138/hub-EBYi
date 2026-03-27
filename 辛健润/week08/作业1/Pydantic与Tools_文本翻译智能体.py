import os
from typing import Optional

import openai
from pydantic import BaseModel, Field


client = openai.OpenAI(
    api_key="sk-69dd3e7fd02e49d3a38498daedaab73d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class TranslationRequest(BaseModel):
    """抽取翻译任务中的关键信息"""

    source_language: str = Field(description="原始语种，需要根据待翻译文本自动识别")
    target_language: str = Field(description="目标语种，需要根据用户要求抽取")
    text: str = Field(description="待翻译的原始文本")


class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def _build_tools(self, response_model: type[BaseModel]):
        schema = response_model.model_json_schema()
        return [
            {
                "type": "function",
                "function": {
                    "name": schema["title"],
                    "description": schema["description"],
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema["required"],
                    },
                },
            }
        ]

    def extract(self, user_prompt: str) -> Optional[TranslationRequest]:
        messages = [
            {
                "role": "system",
                "content": "你是一个信息抽取助手。请从用户请求中抽取原始语种、目标语种和待翻译文本。"
                "如果原始语种没有明确说明，请根据待翻译文本自动识别。",
            },
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self._build_tools(TranslationRequest),
            tool_choice="auto",
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return TranslationRequest.model_validate_json(arguments)
        except Exception:
            print("ERROR", response.choices[0].message)
            return None

    def translate(self, user_prompt: str):
        translation_request = self.extract(user_prompt)
        if translation_request is None:
            return None

        messages = [
            {
                "role": "system",
                "content": "你是一个专业翻译助手。请严格按照目标语种翻译，并且只返回译文本身。",
            },
            {
                "role": "user",
                "content": (
                    f"请将下面的{translation_request.source_language}文本翻译成"
                    f"{translation_request.target_language}：{translation_request.text}"
                ),
            },
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        translated_text = response.choices[0].message.content.strip()

        return {
            "source_language": translation_request.source_language,
            "target_language": translation_request.target_language,
            "text": translation_request.text,
            "translation": translated_text,
        }


if __name__ == "__main__":
    agent = TranslationAgent(model_name="qwen-plus")

    user_prompt = "帮我将good！翻译为中文"
    result = agent.translate(user_prompt)

    if result is not None:
        print(f"原始语种：{result['source_language']}")
        print(f"目标语种：{result['target_language']}")
        print(f"待翻译的文本：{result['text']}")
        print(f"翻译结果：{result['translation']}")
