import asyncio
import os
from openai import OpenAI

# --- 1. 配置千问 API ---
# 请在这里填入你的千问 API Key (通常以 sk- 开头)
QWEN_API_KEY = "sk-f7ca59742f714a39b5db8eafc7e369c8"

# 初始化 OpenAI 客户端，但指向阿里云的服务器
client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# --- 2. 定义模型名称 ---
# 千问常用的模型：qwen-plus, qwen-turbo, qwen-max
MODEL_NAME = "qwen-plus"


# --- 3. 定义 Agent 的指令 (Prompt) ---
# 因为不再使用 agents 库，需要简单的函数来模拟 Agent 的行为

def call_llm(system_prompt, user_input):
    """调用千问模型的通用函数"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 调用千问失败: {e}"


# --- 4. 定义具体的业务逻辑 ---

def analyze_sentiment(text):
    prompt = "你是一个情感分析专家。请分析用户输入的文本情感（正面、负面、中性），并给出简短理由。"
    return call_llm(prompt, text)


def extract_entities(text):
    prompt = "你是一个实体识别专家。请从用户输入的文本中提取关键实体（如人名、地名、组织、时间等），并以JSON格式输出。"
    return call_llm(prompt, text)


def route_intent(text):
    prompt = """
    你是一个智能路由助手。请分析用户的输入，判断其意图属于以下哪一类：
    1. 情感分析 (例如：这句话好开心、我很生气、评价一下...) -> 输出: sentiment
    2. 实体提取 (例如：提取人名、找出地名、有哪些组织...) -> 输出: entity

    规则：
    - 只输出 'sentiment' 或 'entity' 这两个单词之一。
    - 不要输出标点符号或其他解释。
    """
    return call_llm(prompt, text)


# --- 5. 主程序循环 ---

async def main():
    print(f"🤖 千问智能路由系统已启动 (模型: {MODEL_NAME})")
    print("👤 输入 'quit' 退出")

    while True:
        user_input = input("\n👤 用户: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if not user_input.strip():
            continue

        # 1. 路由判断
        print("🔍 正在分析意图...")
        intent = route_intent(user_input).strip().lower()
        print(f"🔍 系统识别意图: [{intent}]")

        # 2. 执行任务
        result = ""
        if "sentiment" in intent:
            print("⚡ 正在调用 [情感分析]...")
            result = analyze_sentiment(user_input)
        elif "entity" in intent:
            print("⚡ 正在调用 [实体识别]...")
            result = extract_entities(user_input)
        else:
            print(f"❌ 无法识别意图 (识别结果: {intent})。请明确说明需要情感分析还是实体识别。")
            continue

        # 3. 输出结果
        print(f"✅ 千问回答:\n{result}")


if __name__ == "__main__":
    # 这里的 asyncio.run 主要是为了保持和原代码结构一致
    # 实际上上面的函数是同步的，直接运行也可以
    asyncio.run(main())