import os

os.environ["OPENAI_API_KEY"] = "sk-aaf17dcb8fa140ccbc455d66b2e80205"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from typing import Optional
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


# ======================
# 1. 守卫：判断是不是情感/实体任务
# ======================
class TaskCheckOutput(BaseModel):
    is_allowed_task: bool  # 是否是情感/实体任务


guard_check_agent = Agent(
    name="Guard Check Agent",
    model="qwen3-max-2026-01-23",
    instructions="""
    你只判断用户输入是否是以下两种任务之一：
    1. 情感分类
    2. 实体识别
    是 → is_allowed_task = true
    否 → is_allowed_task = false
    输出严格JSON格式
    """,
    output_type=TaskCheckOutput,
)

# ======================
# 2. 子Agent 1：情感分类
# ======================
sentiment_agent = Agent(
    name="Sentiment Agent",
    model="qwen3-max-2026-01-23",
    handoff_description="情感分类专家，判断正面、负面、中性",
    instructions="你只做情感分类，输出：正面/负面/中性",
)

# ======================
# 3. 子Agent 2：实体识别
# ======================
entity_agent = Agent(
    name="Entity Agent",
    model="qwen3-max-2026-01-23",
    handoff_description="实体识别专家，提取人名、地名、机构名",
    instructions="你只做实体识别，提取人名、地名、机构名",
)


# ======================
# 4. 守卫函数
# ======================
async def task_guardrail(ctx, agent, input_data):
    result = await Runner.run(guard_check_agent, input_data, context=ctx.context)
    output = result.final_output_as(TaskCheckOutput)

    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=not output.is_allowed_task,
    )


# ======================
# 5. 主Agent：路由选择
# ======================
main_agent = Agent(
    name="Main Agent",
    model="qwen3-max-2026-01-23",
    instructions="判断用户任务：情感分类 → 交给Sentiment Agent；实体识别 → 交给Entity Agent",
    handoffs=[sentiment_agent, entity_agent],
    input_guardrails=[InputGuardrail(guardrail_function=task_guardrail)],
)


# ======================
# 主运行
# ======================
async def main():
    print("=" * 60)

    # 测试1：情感
    query1 = "我今天壮怀激烈"
    print("用户输入：", query1)
    try:
        res = await Runner.run(main_agent, query1)
        print("【情感分类结果】", res.final_output)
    except Exception as e:
        print("拒绝：", e)

    print("-" * 60)

    # 测试2：实体
    query2 = "小明在湖南大学读书，住在长沙"
    print("用户输入：", query2)
    try:
        res = await Runner.run(main_agent, query2)
        print("【实体识别结果】", res.final_output)
    except Exception as e:
        print("拒绝：", e)

    print("-" * 60)

    # 测试3：无关任务（会被拦截）
    query3 = "巴拉拉巴巴,欢迎来到麦当劳"
    print("用户输入：", query3)
    try:
        res = await Runner.run(main_agent, query3)
        print(res.final_output)
    except Exception as e:
        print("【系统拦截】不是情感/实体任务")


if __name__ == "__main__":
    asyncio.run(main())