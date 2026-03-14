import openai # 调用大模型
import json

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-1f98e005abc84b42a0bf3ea03f001d52",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": """你是一个专门做中文意图识别与槽位抽取的信息抽取助手。
输入：一条中文用户语句（短句或问句）。
输出要求（必须严格遵守）：
1) 只返回一个有效的 JSON 对象，且不得包含任何额外说明文字、代码块或注释。
2) JSON 必含三个字段：
   - "domain": 字符串，取值为给定领域列表中的一项；如果无法判定请填 "NaN"。
   - "intent": 字符串，取值为给定意图列表中的一项；如果无法判定请填 "NaN"。
   - "slots": 对象（map），键为实体类型，值为实体数组，每个实体为对象：{"text":"实体原文","start":起始字符索引,"end":结束字符索引}。字符索引为 UTF-8 字符位置，0-based，end 为包含端点。若无实体返回空对象 {}。
3) 若有多处同类型实体，放在同一类型数组中。
4) 列表（参考）：
   - 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
   - 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
   - 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time
5) 如果无法确定 domain/intent 或没有任何 slot，请使用 "NaN" 或空对象 {}。不要猜测额外字段。
6) 输出示例（必须严格 JSON）：
{
  "domain": "cookbook",
  "intent": "QUERY",
  "slots": {
    "dishName": [{"text":"糖醋鲤鱼","start":0,"end":3}]
  }
}
现在仅根据后续的用户输入生成符合上面规则的 JSON。"""},

        {"role": "user", "content": "帮我导航到从北京到郑州到武汉"},

    ],
)
result = completion.choices
print(result, '查看生成的结果')

"""
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
"""