"""
用户输入: 今天心情不好。
--------------------------------------------------
【情感分析结果】
  情感标签: negative
  置信度: 0.95
  解释: 文本中明确使用'心情不好'直接表达负面情绪，无积极或中立词汇，情感倾向清晰。
--------------------------------------------------
"""


import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-14ddbf6d3e1c41c5ae4e9088e9c6dbfc"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from typing import Optional,List
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, handoff
from agents.exceptions import InputGuardrailTripwireTriggered
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class SentimentResult(BaseModel):
    """情感分类结果模型"""
    label: str          # 情感标签: positive, negative, neutral
    confidence: float   # 置信度，范围 0-1
    explanation: str    # 简要解释


class Entity(BaseModel):
    """单个实体模型"""
    entity_type: str    # 实体类型，如 PERSON, ORG, LOC, DATE 等
    text: str           # 实体文本
    position: int       # 在原文中的起始位置


class NERResult(BaseModel):
    """实体识别结果模型"""
    entities: List[Entity]
    summary: str        # 实体摘要描述

sentiment_agent = Agent(
    name="SentimentAnalysisAgent",
    model="qwen3.5-plus",
    instructions="""你是一个情感分析专家。分析用户输入的文本情感，输出情感分类结果。

情感标签只能是以下三种之一：
- positive: 正面/积极的情感
- negative: 负面/消极的情感
- neutral: 中立/无明显情感倾向

请提供置信度分数（0-1之间）和简要的解释说明。

【输出要求】你必须以 JSON 格式输出，包含 label、confidence、explanation 三个字段。""",
    output_type=SentimentResult,
)


ner_agent = Agent(
    name="NERAgent",
    model="qwen3.5-plus",
    instructions="""你是一个命名实体识别专家。从用户输入的文本中提取所有命名实体，包括但不限于：
- PERSON: 人名
- ORG: 组织机构名
- LOC: 地理位置
- DATE: 日期时间
- MISC: 其他类型实体

每个实体需要标注类型、原文文本和位置索引（字符起始位置）。
最后提供一段摘要说明识别出了哪些实体。

【输出要求】你必须以 JSON 格式输出，包含 entities 数组和 summary 字段。entities 中每个元素需包含 entity_type、text、position。""",
    output_type=NERResult,
)


# ==================== 创建主代理 ====================
triage_agent = Agent(
    name="TriageAgent",
    model="qwen3.5-plus",
    instructions="""你是一个任务协调者。根据用户的输入内容，判断应该执行什么任务：

1. 如果用户请求进行情感分析（如：判断情感、分析情绪、这个评论是好评还是差评等），
   请使用 transfer_to_sentiment_agent 将任务移交给情感分析专家代理。

2. 如果用户请求进行实体识别（如：提取人名、识别实体、找出关键信息中的实体等），
   请使用 transfer_to_ner_agent 将任务移交给实体识别专家代理。

""",
    handoffs=[
        handoff(sentiment_agent),
        handoff(ner_agent),
    ],
)


async def run_agent(user_input: str):
    """运行主代理，处理用户请求"""
    print(f"\n用户输入: {user_input}")
    print("-" * 50)

    # 运行主代理
    result = await Runner.run(triage_agent, user_input)

    # 根据最后处理的代理输出相应结果
    if result.last_agent.name == "SentimentAnalysisAgent":
        print("【情感分析结果】")
        sentiment = result.final_output_as(SentimentResult)
        print(f"  情感标签: {sentiment.label}")
        print(f"  置信度: {sentiment.confidence:.2f}")
        print(f"  解释: {sentiment.explanation}")

    elif result.last_agent.name == "NERAgent":
        print("【实体识别结果】")
        ner = result.final_output_as(NERResult)
        if ner.entities:
            print("  识别到的实体:")
            for entity in ner.entities:
                print(f"    - [{entity.entity_type}] {entity.text}")
        else:
            print("  未识别到任何实体")
        print(f"  摘要: {ner.summary}")

    else:
        print("【主代理响应】")
        print(result.final_output)

    print("-" * 50)
    return result


async def main():
    print("=" * 50)
    print("多代理系统演示")
    print("=" * 50)
    await run_agent("今天心情不好。")


if __name__ == "__main__":
    asyncio.run(main())
