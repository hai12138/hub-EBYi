import openai
from pydantic import BaseModel, Field

client = openai.OpenAI(
    api_key="sk-36b1045d11fa4710967a1f6e71c4efce",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "required": response_model.model_json_schema()['required'], # 必填参数
                    }
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print(e)
            return None

class TranslationAgent(BaseModel):
    """
    根据用户输入的文本，判定需要翻译的文本（原始语种、目标语种，待翻译的文本）
    """
    original_language: str = Field(..., description="原始语种")
    target_language: str = Field(..., description="目标语种")
    text: str = Field(..., description="待翻译的文本")

result = ExtractionAgent(model_name="qwen-plus").call("帮我将good！翻译为中文", TranslationAgent)
print(result)
