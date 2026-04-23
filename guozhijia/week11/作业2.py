import os
import asyncio
import uuid

# 配置API
os.environ["OPENAI_API_KEY"] = "sk-f988e27a26444967885d84f2326d4f83d"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents import Agent, Runner, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, ModelSettings
from agents import set_default_openai_api, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from agents import RawResponsesStreamEvent, TResponseInputItem, trace

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL = "qwen3.5-flash"


# ========== 自定义Tool 1: 计算器 ==========
@function_tool
def calculate(expression: str) -> str:
    """计算数学表达式的结果，支持加减乘除、幂运算等基本数学运算。
    Args:
        expression: 数学表达式，如 '2+3*4' 或 '(10+5)/3'
    """
    try:
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不安全字符，仅支持数字和基本运算符(+-*/().%)"
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误：{str(e)}"


# ========== 自定义Tool 2: BMI计算 ==========
@function_tool
def calculate_bmi(weight: float, height: float) -> str:
    """根据体重和身高计算BMI（身体质量指数），并给出健康评估。
    Args:
        weight: 体重(公斤)
        height: 身高(厘米)
    """
    try:
        height_m = height / 100.0
        if height_m <= 0 or weight <= 0:
            return "错误：身高和体重必须为正数"
        bmi = weight / (height_m ** 2)
        bmi = round(bmi, 2)

        if bmi < 18.5:
            category = "偏瘦"
        elif bmi < 24:
            category = "正常"
        elif bmi < 28:
            category = "偏胖"
        else:
            category = "肥胖"

        return f"BMI值: {bmi}，健康评估: {category}（偏瘦<18.5, 正常18.5-24, 偏胖24-28, 肥胖≥28）"
    except Exception as e:
        return f"计算错误：{str(e)}"


# ========== 自定义Tool 3: 体脂率计算 ==========
@function_tool
def calculate_body_fat(gender: str, height: float, weight: float, age: int) -> str:
    """根据性别、身高、体重和年龄估算体脂率（使用BMI法），并给出评估。
    Args:
        gender: 性别：'男' 或 '女'
        height: 身高(厘米)
        weight: 体重(公斤)
        age: 年龄
    """
    try:
        height_m = height / 100.0
        if height_m <= 0 or weight <= 0 or age <= 0:
            return "错误：身高、体重和年龄必须为正数"

        bmi = weight / (height_m ** 2)

        if gender == "男":
            body_fat = 1.2 * bmi + 0.23 * age - 16.2
        elif gender == "女":
            body_fat = 1.2 * bmi + 0.23 * age - 5.4
        else:
            return "错误：性别请输入 '男' 或 '女'"

        body_fat = round(body_fat, 2)

        if gender == "男":
            if body_fat < 10:
                category = "偏瘦"
            elif body_fat < 20:
                category = "正常"
            elif body_fat < 25:
                category = "偏胖"
            else:
                category = "肥胖"
        else:
            if body_fat < 18:
                category = "偏瘦"
            elif body_fat < 28:
                category = "正常"
            elif body_fat < 33:
                category = "偏胖"
            else:
                category = "肥胖"

        return f"体脂率: {body_fat}%，健康评估: {category}（男：偏瘦<10%, 正常10-20%, 偏胖20-25%, 肥胖≥25%；女：偏瘦<18%, 正常18-28%, 偏胖28-33%, 肥胖≥33%）"
    except Exception as e:
        return f"计算错误：{str(e)}"


# ========== 创建Agent ==========
external_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

agent = Agent(
    name="enterprise_assistant",
    model=OpenAIChatCompletionsModel(
        model=MODEL,
        openai_client=external_client,
    ),
    instructions="你是一个企业职能助手，可以帮助员工进行日常计算和健康评估。你可以使用计算器、BMI计算和体脂率计算工具来帮助用户。请用中文回答。",
    tools=[calculate, calculate_bmi, calculate_body_fat],
    model_settings=ModelSettings(parallel_tool_calls=False),
)


async def run_test(prompt: str):
    """运行单个测试"""
    print(f"用户输入: {prompt}")
    print("-" * 40)

    result = Runner.run_streamed(
        agent,
        input=prompt,
        run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False)),
    )

    tool_calls = []
    final_output = ""

    async for event in result.stream_events():
        if event.type == "raw_response_event" and hasattr(event, 'data'):
            if isinstance(event.data, ResponseOutputItemDoneEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    tool_calls.append({
                        "name": event.data.item.name,
                        "arguments": event.data.item.arguments,
                    })
            if isinstance(event.data, ResponseTextDeltaEvent):
                final_output += event.data.delta
                print(event.data.delta, end="", flush=True)

    print()

    # 打印工具调用详情
    if tool_calls:
        print()
        print("[工具调用详情]")
        for tc in tool_calls:
            print(f"  工具名称: {tc['name']}")
            print(f"  调用参数: {tc['arguments']}")

    return final_output, tool_calls


async def main():
    print("=" * 60)
    print("企业职能助手 - 自定义工具调用测试")
    print("=" * 60)
    print()

    # 测试1: 计算器
    print("=" * 60)
    print("测试1: 计算器工具")
    print("=" * 60)
    await run_test("帮我算一下 (125 + 378) * 23 等于多少？")
    print()

    # 测试2: BMI计算
    print("=" * 60)
    print("测试2: BMI计算工具")
    print("=" * 60)
    await run_test("我身高175厘米，体重70公斤，帮我算一下BMI")
    print()

    # 测试3: 体脂率计算
    print("=" * 60)
    print("测试3: 体脂率计算工具")
    print("=" * 60)
    await run_test("我是男性，30岁，身高178厘米，体重82公斤，帮我计算体脂率")
    print()


if __name__ == "__main__":
    asyncio.run(main())
