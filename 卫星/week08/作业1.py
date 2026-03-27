from pydantic import BaseModel, Field
import openai


# =========================
# 1. 初始化客户端
# =========================
client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# =========================
# 2. 通用信息抽取智能体
# =========================
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt: str, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        schema = response_model.model_json_schema()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema["title"],           # 工具名字
                    "description": schema["description"],  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],   # 参数说明
                        "required": schema["required"],       # 必须字段
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print("ERROR:", e)
            print("模型返回内容：", response.choices[0].message)
            return None


# =========================
# 3. 定义“翻译任务”结构
# =========================
class TranslationTask(BaseModel):
    """自动识别翻译任务中的原始语种、目标语种和待翻译文本"""
    source_language: str = Field(description="原始语种，例如中文、英文、日文、韩文等；如果用户未明确说明，可根据待翻译文本自动判断")
    target_language: str = Field(description="目标语种，例如中文、英文、日文、韩文等")
    text: str = Field(description="需要翻译的原始文本内容")


# =========================
# 4. 翻译智能体
# =========================
class TranslationAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.extractor = ExtractionAgent(model_name)

    def extract_task(self, user_prompt: str):
        return self.extractor.call(user_prompt, TranslationTask)

    def translate(self, task: TranslationTask):
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的翻译助手，请准确翻译文本，保持原意，不要添加任何多余解释。"
            },
            {
                "role": "user",
                "content": f"请将以下内容从{task.source_language}翻译成{task.target_language}：\n{task.text}"
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        return response.choices[0].message.content

    def call(self, user_prompt: str):
        task = self.extract_task(user_prompt)
        if task is None:
            return None

        translation = self.translate(task)

        return {
            "source_language": task.source_language,
            "target_language": task.target_language,
            "text": task.text,
            "translation": translation
        }


# =========================
# 5. 测试
# =========================
if __name__ == "__main__":
    agent = TranslationAgent(model_name="qwen-plus")

    test_cases = [
        "帮我将good!翻译为中文",
        "请把“我今天很开心”翻译成英文",
        "把 this is a beautiful day 翻译成中文",
    ]

    for text in test_cases:
        result = agent.call(text)
        print("用户输入：", text)
        print("输出结果：", result)
        print("-" * 60)
