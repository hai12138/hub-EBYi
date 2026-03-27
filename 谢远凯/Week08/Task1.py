from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal

import openai
import json

# 初始化OpenAI兼容客户端（对接阿里云通义千问）
client = openai.OpenAI(
    api_key="sk-bf48a63789aa43df98eaaa1873b279d9",  # 替换为你的API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ===================== 核心抽取智能体类 =====================
class ExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        """
        调用大模型进行结构化信息抽取
        :param user_prompt: 用户输入的自然语言文本
        :param response_model: Pydantic模型，定义抽取格式
        :return: 结构化的抽取结果（Pydantic实例）或None
        """
        # 构造对话消息
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # 自动生成工具调用配置
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
                    },
                }
            }
        ]

        try:
            # 调用大模型
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # 强制调用工具
                temperature=0.1,  # 降低随机性，保证抽取准确性
            )

            # 解析工具调用结果
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                print("ERROR: 模型未返回工具调用结果")
                return None

            # 提取并校验参数
            arguments = tool_calls[0].function.arguments
            result = response_model.model_validate_json(arguments)
            return result

        except KeyError as e:
            print(f"ERROR: 缺少关键字段 - {e}")
            return None
        except json.JSONDecodeError:
            print("ERROR: 返回参数不是合法的JSON格式")
            return None
        except Exception as e:
            print(f"ERROR: 未知错误 - {e}")
            return None


# ===================== 翻译数据模型定义 =====================
class TranslationRequest(BaseModel):
    """
    文本翻译请求参数抽取
    用于从用户输入中抽取翻译所需的关键信息
    """
    # 原始语种（支持常见语种，默认自动识别）
    source_language: Literal["自动识别", "中文", "英语", "日语", "韩语", "法语", "德语", "西班牙语"] = Field(
        default="自动识别",
        description="待翻译文本的原始语种，如未指定则为自动识别"
    )
    # 目标语种（必选）
    target_language: Literal["中文", "英语", "日语", "韩语", "法语", "德语", "西班牙语"] = Field(
        description="翻译后的目标语种"
    )
    # 待翻译文本（必选）
    text_to_translate: str = Field(
        description="需要被翻译的文本内容"
    )


# ===================== 翻译执行类 =====================
class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
        self.extraction_agent = ExtractionAgent(model_name)  # 复用抽取智能体

    def translate(self, user_prompt):
        """
        完整翻译流程：抽取翻译参数 → 执行翻译 → 返回结果
        :param user_prompt: 用户输入的翻译请求（自然语言）
        :return: 包含抽取信息和翻译结果的字典
        """
        # 第一步：抽取翻译所需的结构化信息
        extract_result = self.extraction_agent.call(user_prompt, TranslationRequest)
        if not extract_result:
            return {"status": "failed", "message": "抽取翻译参数失败"}

        # 第二步：构造翻译提示词
        translate_prompt = f"""
        请你完成以下翻译任务：
        - 原始语种：{extract_result.source_language}
        - 目标语种：{extract_result.target_language}
        - 待翻译文本：{extract_result.text_to_translate}

        请只返回翻译结果，不要添加任何额外解释。
        """

        # 第三步：调用大模型执行翻译
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": translate_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            translation_result = response.choices[0].message.content.strip()

            # 整合结果
            final_result = {
                "status": "success",
                "extraction_info": {
                    "source_language": extract_result.source_language,
                    "target_language": extract_result.target_language,
                    "text_to_translate": extract_result.text_to_translate
                },
                "translation_result": translation_result
            }
            return final_result

        except Exception as e:
            return {"status": "failed", "message": f"翻译执行失败：{str(e)}"}


# ===================== 测试用例 =====================
if __name__ == "__main__":
    # 初始化翻译智能体
    translator = TranslationAgent(model_name="qwen-plus")

    # 测试用例1：基础翻译（英语→中文）
    test_case_1 = "帮我将very good！翻译为中文"
    result_1 = translator.translate(test_case_1)
    print("=" * 60)
    print(f"测试用例1输入：{test_case_1}")
    if result_1["status"] == "success":
        print(f"抽取信息：")
        print(f"原始语种：{result_1['extraction_info']['source_language']}")
        print(f"目标语种：{result_1['extraction_info']['target_language']}")
        print(f"待翻译文本：{result_1['extraction_info']['text_to_translate']}")
        print(f"翻译结果：{result_1['translation_result']}")
    else:
        print(f"失败原因：{result_1['message']}")

    # 测试用例2：指定原始语种（日语→英语）
    test_case_2 = "请把「こんにちは」从日语翻译成英语"
    result_2 = translator.translate(test_case_2)
    print("=" * 60)
    print(f"测试用例2输入：{test_case_2}")
    if result_2["status"] == "success":
        print(f"抽取信息：")
        print(f"原始语种：{result_2['extraction_info']['source_language']}")
        print(f"目标语种：{result_2['extraction_info']['target_language']}")
        print(f"待翻译文本：{result_2['extraction_info']['text_to_translate']}")
        print(f"翻译结果：{result_2['translation_result']}")
    else:
        print(f"失败原因：{result_2['message']}")

    # 测试用例3：自动识别语种（法语→中文）
    test_case_3 = "帮我翻译Bonjour为中文"
    result_3 = translator.translate(test_case_3)
    print("=" * 60)
    print(f"测试用例3输入：{test_case_3}")
    if result_3["status"] == "success":
        print(f"抽取信息：")
        print(f"原始语种：{result_3['extraction_info']['source_language']}")
        print(f"目标语种：{result_3['extraction_info']['target_language']}")
        print(f"待翻译文本：{result_3['extraction_info']['text_to_translate']}")
        print(f"翻译结果：{result_3['translation_result']}")
    else:
        print(f"失败原因：{result_3['message']}")
