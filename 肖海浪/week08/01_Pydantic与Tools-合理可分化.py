from pydantic import BaseModel, Field
import openai
import json
import sys

# ================= 1. API 配置 (使用您自己的 Key) =================
# 请确认此 Key 有效，若失效请替换
API_KEY = "sk-d342b662f6ed4ee2af999c15a6b3c549"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ================= 2. 定义 Pydantic 数据模型 =================
class TranslationTask(BaseModel):
    """
    从用户指令中提取翻译任务的结构化信息。
    输出必须包含源语言、目标语言和待翻译的具体文本。
    """
    source_language: str = Field(description="原始语种，例如：英语、中文、日语、法语等")
    target_language: str = Field(description="目标语种，即用户希望翻译成的语言")
    text_to_translate: str = Field(description="需要被翻译的原始文本内容，不包含指令词")


# ================= 3. ExtractionAgent 实现 (参考原文件逻辑) =================
class ExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        """
        调用大模型，利用 Tool/Function Call 机制提取信息并返回 Pydantic 对象
        """
        messages = [
            {
                "role": "system",
                "content": "你是一个精准的信息提取助手。请分析用户输入，提取翻译所需的三个要素。不要输出任何多余的解释，只通过工具调用返回结果。"
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # 构建 Tool 定义
        schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema.get('title', 'extract_translation_info'),
                    "description": schema.get('description', 'Extract translation details'),
                    "parameters": {
                        "type": "object",
                        "properties": schema['properties'],
                        "required": schema.get('required', []),
                    },
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": schema.get('title', 'extract_translation_info')}},
                # 强制调用工具
                temperature=0.1  # 低温度保证稳定性
            )

            # 解析工具调用结果
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                arguments_str = tool_call.function.arguments

                # 将 JSON 字符串转换为 Pydantic 模型
                return response_model.model_validate_json(arguments_str)
            else:
                raise Exception("模型未返回工具调用结果")

        except Exception as e:
            print(f"\n❌ 处理错误: {e}", file=sys.stderr)
            return None


# ================= 4. 主程序：交互式循环 =================
def main():
    agent = ExtractionAgent(model_name="qwen-plus")

    print("=" * 60)
    print("🤖 文本翻译智能体已启动")
    print("请输入翻译指令 (例如: '帮我把 good! 翻译成中文')")
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 60)

    while True:
        try:
            # 获取用户输入
            user_input = input("\n📝 请输入指令: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            # 调用智能体提取信息
            result = agent.call(user_input, TranslationTask)

            if result:
                # 构造最终输出的 JSON 字典
                output_data = {
                    "source_language": result.source_language,
                    "target_language": result.target_language,
                    "text_to_translate": result.text_to_translate
                }

                # 打印漂亮的 JSON 输出
                print("\n✅ 识别结果 (JSON):")
                print(json.dumps(output_data, ensure_ascii=False, indent=2))

                # (可选) 如果您想直接看翻译结果，可以在这里调用翻译接口
                # print(f"\n💡 翻译提示: 请将 '{result.text_to_translate}' 从 {result.source_language} 翻译到 {result.target_language}")
            else:
                print("⚠️ 未能提取到有效信息，请尝试更清晰的指令。")

        except KeyboardInterrupt:
            print("\n👋 程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生未知错误: {e}")


if __name__ == "__main__":
    main()