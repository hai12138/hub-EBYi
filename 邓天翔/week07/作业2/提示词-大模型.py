from openai import OpenAI
import pandas as pd

intents = pd.read_csv('intents.txt', sep='\t', header=None, names=['intent'])
intents_list = intents['intent'].tolist()
print("意图分类:", intents_list)
slots = pd.read_csv('slots.txt', sep='\t', header=None, names=['slot'])
slots_list = slots['slot'].tolist()
print("实体类型:", slots_list)
domains = pd.read_csv('domains.txt', sep='\t', header=None, names=['domain'])
domains_list = domains['domain'].tolist()
print("领域分类:", domains_list)


question = "清蒸鱼怎么做？"



client = OpenAI(
    api_key="sk-2427d35ff6724d30856a786b596fb0bf",

    # 大模型厂商的地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "user", "content":f"""
        # 信息解析智能对话系统

## 角色定义
你是一个专业信息解析智能对话系统，能够理解用户输入的自然语言，准确识别用户的意图并提取关键信息实体。
请帮我对这条信息进行解析："{question}"。

## 任务目标
对于用户输入的每条消息，你需要：
1. 识别用户所属的应用领域（Domain）
2. 识别用户的真实意图（Intent）
3. 提取关键实体信息（Slots）
4. 以结构化JSON格式返回分析结果

## 领域分类（Domain Categories）
以下是系统支持的领域类别：{domains_list}

## 意图分类（Intent Categories）
以下是系统支持的意图类别：{intents_list}

## 实体类型（Slot Types）
以下是需要提取的实体类型：{slots_list}

## 输入输出格式

### 输入
用户输入的自然语言文本

### 输出
返回以下JSON格式的结果：
{{
  "text": "用户输入的原始文本",
  "domain": "识别的应用领域",
  "intent": "识别的意图类别",
  "slots": {{
    "entity_type": "提取的实体值"
  }}
}}

## 示例

示例1:
输入："请帮我打开uc"
输出：
{{
  "text": "请帮我打开uc",
  "domain": "app",
  "intent": "LAUNCH",
  "slots": {{
    "name": "uc"
  }}
}}

示例2:
输入："播放周杰伦的歌"
输出：
{{
  "text": "播放周杰伦的歌",
  "domain": "music",
  "intent": "PLAY",
  "slots": {{
    "artistRole": "周杰伦",
    "media": "歌"
  }}
}}

示例3:
输入："查询明天天气"
输出：
{{
  "text": "查询明天天气",
  "domain": "weather",
  "intent": "QUERY",
  "slots": {{
    "datetime_date": "明天"
  }}
}}

示例4:
输入："我想看新闻"
输出：
{{
  "text": "我想看新闻",
  "domain": "news",
  "intent": "SEARCH",
  "slots": {{}}
}}

## 注意事项
1. 领域识别要准确反映用户请求所属的应用领域
2. 意图识别要准确反映用户的真实需求
3. 实体提取要完整且准确
4. 当无法确定领域时，根据意图内容推测最合适的领域
5. 当无法确定意图时，使用"DEFAULT"
6. 实体类型必须从定义的列表中选择
7. 如果没有匹配的实体，slots字段为空对象{{}}
8. 保持输出JSON格式的完整性

        """}
    ]
)
print(completion.choices[0].message.content)
