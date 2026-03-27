from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal
import openai
import json

# 初始化 OpenAI 客户端（兼容阿里云百炼）
client = openai.OpenAI(
    api_key="sk-411bcb4cd2a940f69a02a6d239b9ec4",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ===================== 第一步：定义结构化抽取模型 =====================
class TranslationInfo(BaseModel):
    """文本翻译信息抽取：识别原始语种、目标语种、待翻译文本"""
    # 定义常见语种（可根据需要扩展）
    source_language: Literal["中文", "英语", "日语", "法语", "德语", "西班牙语"] = Field(
        description="待翻译文本的原始语种（如：英语、中文）"
    )
    target_language: Literal["中文", "英语", "日语", "法语", "德语", "西班牙语"] = Field(
        description="需要翻译成的目标语种（如：中文、英语）"
    )
    text_to_translate: str = Field(
        description="需要翻译的原始文本内容"
    )


# ===================== 第二步：定义包含翻译结果的完整模型 =====================
class TranslationResult(BaseModel):
    """翻译结果完整信息"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    text_to_translate: str = Field(description="待翻译文本")
    translated_text: str = Field(description="翻译后的文本结果")


# ===================== 第三步：增强版翻译智能体 =====================
class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def _extract_translation_info(self, user_prompt) -> TranslationInfo:
        """内部方法：抽取翻译所需的结构化信息"""
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 自动生成 Tools 格式（基于 Pydantic 模型的 Schema）
        tools = [
            {
                "type": "function",
                "function": {
                    "name": TranslationInfo.model_json_schema()['title'],
                    "description": TranslationInfo.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": TranslationInfo.model_json_schema()['properties'],
                        "required": TranslationInfo.model_json_schema()['required'],
                    },
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # 让模型自动选择调用函数
            )
            # 提取函数调用的参数
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            # 验证并转换为 Pydantic 模型
            return TranslationInfo.model_validate_json(arguments)
        except Exception as e:
            print(f'抽取翻译信息失败：{e}')
            raise ValueError("无法识别你的翻译需求，请明确说明原始语种、目标语种和待翻译文本")

    def _translate_text(self, translation_info: TranslationInfo) -> str:
        """内部方法：调用大模型完成实际翻译"""
        # 构造翻译提示词（清晰明确，提升翻译准确性）
        translate_prompt = f"""
        请将以下{translation_info.source_language}文本翻译成{translation_info.target_language}，仅返回翻译结果，不要额外解释：
        待翻译文本：{translation_info.text_to_translate}
        """

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": translate_prompt}
                ],
                temperature=0.1,  # 低温度保证翻译准确性
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f'翻译执行失败：{e}')
            raise ValueError("翻译服务暂时不可用，请稍后重试")

    def translate(self, user_prompt) -> TranslationResult:
        """核心方法：一站式完成「信息抽取 + 翻译」"""
        # 1. 抽取翻译信息
        translation_info = self._extract_translation_info(user_prompt)
        # 2. 执行翻译
        translated_text = self._translate_text(translation_info)
        # 3. 组装完整结果并返回
        return TranslationResult(
            source_language=translation_info.source_language,
            target_language=translation_info.target_language,
            text_to_translate=translation_info.text_to_translate,
            translated_text=translated_text
        )


# ===================== 测试用例 =====================
if __name__ == "__main__":
    # 初始化翻译智能体
    agent = TranslationAgent(model_name="qwen-plus")

    # 测试用例 1：英文转中文
    prompt1 = "帮我将good！翻译为中文"
    result1 = agent.translate(prompt1)
    print("=== 测试用例 1 ===")
    print(f"原始语种：{result1.source_language}")
    print(f"目标语种：{result1.target_language}")
    print(f"待翻译文本：{result1.text_to_translate}")
    print(f"翻译结果：{result1.translated_text}\n")

    # 测试用例 2：中文转英文
    prompt2 = "请把「我今天很开心」翻译成英语"
    result2 = agent.translate(prompt2)
    print("=== 测试用例 2 ===")
    print(f"原始语种：{result2.source_language}")
    print(f"目标语种：{result2.target_language}")
    print(f"待翻译文本：{result2.text_to_translate}")
    print(f"翻译结果：{result2.translated_text}\n")

    # 测试用例 3：法语转中文
    prompt3 = "将Bonjour le monde翻译成中文"
    result3 = agent.translate(prompt3)
    print("=== 测试用例 3 ===")
    print(f"原始语种：{result3.source_language}")
    print(f"目标语种：{result3.target_language}")
    print(f"待翻译文本：{result3.text_to_translate}")
    print(f"翻译结果：{result3.translated_text}")
