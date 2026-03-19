"""
阿里通义千问大模型 Demo
使用提示词完成智能对话信息解析任务
"""

import json
import os
from http import HTTPStatus
import dashscope


def load_prompt_template():
    """加载提示词模板"""
    template = """你是一个专业的智能对话信息解析助手，能够理解用户的自然语言输入，并准确识别用户的意图和关键信息。

### 你的任务
1. 领域识别（Domain）：判断用户查询属于哪个领域
2. 意图识别（Intent）：识别用户想要执行的操作
3. 槽位填充（Slot Filling）：提取用户输入中的关键信息

### 支持的领域
app, music, weather, map, bus, train, flight, message, contacts, telephone, news, video, radio, stock, lottery, novel, match, website, translation, tvchannel, cinemas, cookbook, joke, riddle, poetry, epg, health, email, story

### 支持的意图
LAUNCH, OPEN, SEARCH, QUERY, PLAY, DOWNLOAD, DIAL, SEND, REPLY, FORWARD, CREATE, VIEW, ROUTE, POSITION, LOOK_BACK, REPLAY_ALL, TRANSLATION, NUMBER_QUERY, DATE_QUERY, CLOSEPRICE_QUERY, RISERATE_QUERY, SENDCONTACTS, DEFAULT

### 常见槽位类型
name, Src, Dest, artist, song, date, location_city, content, keyword, receiver, startLoc_city, endLoc_city, startDate_date, startDate_time

### 输出格式
请严格按照以下JSON格式输出，不要添加任何其他文字：
{
  "text": "用户原始输入",
  "domain": "识别的领域",
  "intent": "识别的意图",
  "slots": {
    "槽位名": "槽位值"
  }
}

### 示例
输入：打开微信
输出：{"text": "打开微信", "domain": "app", "intent": "LAUNCH", "slots": {"name": "微信"}}

输入：播放周杰伦的稻香
输出：{"text": "播放周杰伦的稻香", "domain": "music", "intent": "PLAY", "slots": {"artist": "周杰伦", "song": "稻香"}}

输入：从北京到上海怎么坐车
输出：{"text": "从北京到上海怎么坐车", "domain": "bus", "intent": "QUERY", "slots": {"Src": "北京", "Dest": "上海"}}
"""
    return template


def call_qwen(user_input, api_key=None):
    """调用通义千问API"""

    # 设置API Key
    if api_key:
        dashscope.api_key = api_key
    else:
        # 从环境变量读取
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

    if not dashscope.api_key:
        return {"error": "请设置DASHSCOPE_API_KEY环境变量或传入api_key参数"}

    # 构建提示词
    prompt_template = load_prompt_template()
    full_prompt = f"{prompt_template}\n\n现在请解析：{user_input}"

    # 调用API
    try:
        response = dashscope.Generation.call(
            model='qwen-qwen-plus',
            prompt=full_prompt,
            result_format='message',
            max_tokens=500,
            temperature=0.1,
            top_p=0.5
        )

        if response.status_code == HTTPStatus.OK:
            result_text = response.output.choices[0].message.content

            # 提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result_json = json.loads(json_match.group())
                return result_json
            else:
                return {"error": "无法从响应中提取JSON", "raw": result_text}
        else:
            return {
                "error": f"API调用失败: {response.code}",
                "message": response.message
            }

    except Exception as e:
        return {"error": f"调用异常: {str(e)}"}

def simple_test():
    """简单测试"""

    print("=" * 80)
    print("阿里通义千问 - 简单测试")
    print("=" * 80)

    # 测试用例
    test_cases = [
        "打开微信",
        "播放周杰伦的稻香",
        "从北京到上海怎么坐车",
        "明天北京天气怎么样",
        "给张三发短信说晚上一起吃饭"
    ]

    for i, user_input in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试 {i}/{len(test_cases)}")
        print(f"{'='*80}")
        print(f"输入: {user_input}")

        result = call_qwen(user_input)

        if "error" in result:
            print(f" 错误: {result['error']}")
        else:
            print(f" 结果: {json.dumps(result, ensure_ascii=False)}")

        # 避免频繁调用
        if i < len(test_cases):
            import time
            time.sleep(1)


if __name__ == "__main__":
     simple_test()
