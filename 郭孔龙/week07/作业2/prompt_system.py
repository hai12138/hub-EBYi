# 创建文件：prompt_system.py

class DialogueSystemPrompt:
    def __init__(self):
        self.system_prompt = """你是一个专业的智能对话语义理解系统（Semantic Understanding System）。
你的任务是准确理解用户输入的语义，提取关键信息，为后续的任务执行提供支持。

## 核心能力
1. 意图识别：判断用户想要做什么
2. 槽位填充：提取关键实体信息
3. 领域分类：判断属于哪个服务领域
4. 置信度评估：评估预测的可靠性

## 输出要求
1. 格式：严格的 JSON
2. 准确性：尽可能准确
3. 完整性：包含所有必要信息
4. 一致性：使用标准术语

## 支持的主要意图
【应用操作】LAUNCH(打开), QUERY(查询), DOWNLOAD(下载), CLOSE(关闭)
【交通出行】ROUTE(路线), QUERY(查询班次), BOOK(预订)
【媒体播放】PLAY(播放), PAUSE(暂停), SEARCH(搜索)
【通讯】DIAL(拨打), SEND(发送), REPLY(回复)
【生活服务】CREATE(创建), VIEW(查看), BOOK(预订)

## 常见槽位
【名称类】name(应用名), artist(艺术家), song(歌曲), film(电影)
【地点类】Src(出发地), Dest(目的地), location_city(城市)
【时间类】datetime_date(日期), datetime_time(时间), date(日期)
【其他】content(内容), receiver(接收者), keyword(关键词)
"""

        self.few_shot_examples = [
            {
                "input": "打开微信",
                "output": {
                    "domain": "app",
                    "intent": "LAUNCH",
                    "slots": {"name": "微信"},
                    "confidence": 0.98
                }
            },
            {
                "input": "北京到上海的汽车",
                "output": {
                    "domain": "bus",
                    "intent": "QUERY",
                    "slots": {"Src": "北京", "Dest": "上海"},
                    "confidence": 0.95
                }
            },
            {
                "input": "去深圳怎么坐车",
                "output": {
                    "domain": "map",
                    "intent": "ROUTE",
                    "slots": {"endLoc_city": "深圳"},
                    "confidence": 0.93
                }
            },
            {
                "input": "播放周杰伦的稻香",
                "output": {
                    "domain": "music",
                    "intent": "PLAY",
                    "slots": {"artist": "周杰伦", "song": "稻香"},
                    "confidence": 0.97
                }
            },
            {
                "input": "明天早上 8 点叫我起床",
                "output": {
                    "domain": "alarm",
                    "intent": "CREATE",
                    "slots": {"datetime_date": "明天", "datetime_time": "早上 8 点"},
                    "confidence": 0.94
                }
            }
        ]

    def build_prompt(self, user_input: str, use_cot: bool = False) -> str:
        """构建完整的提示词"""

        # 添加示例
        examples_text = ""
        for i, example in enumerate(self.few_shot_examples[:3], 1):
            examples_text += f"### 示例 {i}\n"
            examples_text += f"用户：\"{example['input']}\"\n"
            examples_text += f"输出：{example['output']}\n\n"

        # 思维链提示
        cot_instruction = ""
        if use_cot:
            cot_instruction = """
请先进行逐步思考：
1. 分析用户的核心需求
2. 判断所属领域
3. 确定具体意图
4. 提取关键槽位
5. 评估置信度

思考过程："""

        prompt = f"""{self.system_prompt}

{examples_text}
## 当前任务
用户输入："{user_input}"

{cot_instruction}
请输出 JSON 结果："""

        return prompt

    def parse_response(self, response: str) -> dict:
        """解析模型响应"""
        import json
        import re

        # 提取 JSON 部分
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result
            except:
                return {"error": "JSON 解析失败"}
        return {"error": "未找到 JSON"}


# 使用示例
if __name__ == "__main__":
    prompt_system = DialogueSystemPrompt()

    # 测试
    user_input = "帮我打开 UC 浏览器"
    prompt = prompt_system.build_prompt(user_input, use_cot=True)

    print("=== 提示词 ===")
    print(prompt)
    print("\n=== 将上述提示词发送给大模型 ===")
