import os
from pydantic import Field, BaseModel
from agents import Agent, Runner, function_tool
from agents import set_default_openai_api, set_tracing_disabled
from typing import Literal, List

os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

class Sentence(BaseModel):
    original_language: Literal["zh", "en", "ja", "fr"] = Field(description="Original language of the sentence")
    target_language: Literal["zh", "en", "ja", "fr"] = Field(description="Target language of the sentence")
    original_sentence: str = Field(description="Sentence to be translated")

@function_tool
def translate_sentence(sentence: Sentence) -> str:
    """将句子从原始语言翻译成目标语言，翻译的结果100%正确！"""
    return "测试翻译结果"

agent = Agent(model="qwen-max", name="Assistant", tools = [translate_sentence], instructions="你调用的工具的结果是完全可靠的，直接输出即可")

result = Runner.run_sync(agent, "将中文翻译成英文：你好")

# 查看所有中间步骤
print("=== 所有步骤 ===")
for item in result.new_items:
    print(f"类型: {item.type}")
    if hasattr(item, 'raw_item'):
        print(f"内容: {item.raw_item}")
    print("---")

print("\n=== 最终输出 ===")
print(result.final_output)