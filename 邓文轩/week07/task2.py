import openai # 调用大模型
import json
import os

client = openai.OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",
    n=3,
    temperature=0.3,
    messages=[
        {"role": "system", "content": """你是一个自然语言理解（NLU）专家，擅长领域分类、意图识别和实体抽取。

## 任务说明
对用户输入的文本进行分析，提取出：
- domain: 文本所属领域
- intent: 用户意图
- slots: 提取的实体信息

## 可选值列表

### Domain（领域）
music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story

### Intent（意图）
OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION

### Slots（实体类型）
code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

## 输出要求
1. 严格按照JSON格式输出
2. domain和intent必须从上述列表中选择
3. slots中只包含文本中明确出现的实体
4. 如果文本中没有相关实体，slots应为空对象 {}
5. 实体值必须是文本中的原始内容，不要修改

## 输出格式
```json
{
    "domain": "领域值",
    "intent": "意图值",
    "slots": {
        "实体类型": "实体值"
    }
}
```

## 示例

示例1:
用户输入: 肉肉怎么腌制？
输出:
```json
{
    "domain": "cookbook",
    "intent": "QUERY",
    "slots": {
        "ingredient": "肉肉"
    }
}
```

示例2:
用户输入: 从许昌到中山怎么坐车？
输出:
```json
{
    "domain": "bus",
    "intent": "QUERY",
    "slots": {
        "startLoc_city": "许昌",
        "endLoc_city": "中山"
    }
}
```

示例3:
用户输入: 来一首七言绝句。
输出:
```json
{
    "domain": "poetry",
    "intent": "DEFAULT",
    "slots": {}
}
```

示例4:
用户输入: 中超赛事预告，你。
输出:
```json
{
    "domain": "match",
    "intent": "QUERY",
    "slots": {
      "category": "中超",
      "type": "预告"
    }
}
```

## 注意事项
- 如果文本包含多个意图，选择最主要的意图
- 如果文本不匹配任何已知domain，使用"DEFAULT"
- 如果文本不匹配任何已知intent，使用"DEFAULT"
- 不要输出任何解释性文字，只输出JSON
"""},
        {"role": "user", "content": "给我给我背一首唐诗李白写的《静夜思》"},

    ],
)
result = completion.choices
print(f"返回的choice数量: {len(result)}")  # 输出: 3

# 遍历所有choices
for i, choice in enumerate(result):
    print(f"\nChoice {i+1}:")
    print(choice.message.content)