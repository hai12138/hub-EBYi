import os

os.environ['OPENAI_API_KEY'] = 'sk-594a45dd7c344e4783bf8708fb55c094'
os.environ['OPENAI_BASE_URL'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

import asyncio
from pydantic import BaseModel
from typing import Optional
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 情感分类专家代理
sentimentClassification_tutor_agent = Agent(
    name="SentimentClassification Tutor",
    model="qwen3.5-plus",
    handoff_description="负责处理所有情感分类的专家代理",
    instructions="你是情感分类专家，请根据用户的输入，判断输入的文本情感是正面、负面还是中性",
)
# 实体识别专家代理
entityRecognition_tutor_agent = Agent(
    name="EntityRecognition Tutor",
    model="qwen3.5-plus",
    handoff_description="负责处理所有实体识别的专家代理",
    instructions="你是实体识别专家，请根据用户的输入，判断输入的文本中包含哪些实体，例如人物、组织、地点等。",

)

class TaskOutput(BaseModel):
    """用于判断用户请求是否为情感分类或实体识别任务的结构"""
    is_sentimentClassification_or_entityRecognition_task: bool

guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen3.5-plus",
    instructions="判断用户的问题是否属于情感分类或实体识别任务。如果是，'is_sentimentClassification_or_entityRecognition_task'应为 True， json 返回",
    output_type=TaskOutput, # openai 官方推荐的一个语法，推荐大模型输出的格式类型，国内模型支持的不太好；
)

async def task_guardrail(ctx, agent, input_data):
    """
    运行检查代理来判断输入是否为情感分类或实体识别任务。
    如果不是情感分类或实体识别任务 ('is_sentimentClassification_or_entityRecognition_task' 为 False)，则触发阻断 (tripwire)。
    """
    print(f"\n[Guardrail Check] 正在检查输入: '{input_data}'...")

    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 解析输出
    final_output = result.final_output_as(TaskOutput)

    tripwire_triggered = not final_output.is_sentimentClassification_or_entityRecognition_task

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered,
    )

# 分发代理，意图识别专家代理
triage_agent = Agent(
    name="triage_agent",
    model="qwen3.5-plus",
    instructions="你的任务是根据用户的输入，判断应该将请求分派给'SentimentClassification Tutor'还是'EntityRecognition Tutor'",
    handoffs=[sentimentClassification_tutor_agent, entityRecognition_tutor_agent],
    # input_guardrails=[InputGuardrail(task_guardrail)]
)

async def main():
    print("--- 启动中文代理系统示例 ---")

    print("\n" + "=" * 50)
    print("=" * 50)

    try:
        query = "我今天非常开心"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    print("=" * 50)

    try:
        query = "小王去年在成都上班"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    print("=" * 50)

    try:
        query = "你好"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

if __name__ == "__main__":
    asyncio.run(main())

    try:
        draw_graph(triage_agent, filename="openai-agent-work")
    except:
        print("绘制agent失败，默认跳过。。。")


