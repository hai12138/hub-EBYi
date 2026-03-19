import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "<KEY>"),
    base_url="http://localhost:11434/v1/"
)

SYSTEM_PROMPT = """
你是信息解析专家，需要完成意图识别和命名实体识别任务。

请对用户输入文本进行解析，并且只返回一个合法 JSON 对象，不要返回任何解释、说明、代码块标记或其他非 JSON 内容。

要求：
1. text 字段必须原样返回输入文本
2. domain 必须从以下标签中选择一个：
music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
3. intent 必须从以下标签中选择一个：
OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
4. slots 必须是一个 JSON 对象
5. slots 中的 key 必须从以下标签中选择：
code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time
6. slots 中的 value 必须直接来自原始文本，不允许编造
7. 如果没有可抽取实体，slots 返回空对象 {}

返回格式示例：
{
    "text": "小鸡炖蘑菇的做法",
    "domain": "cookbook",
    "intent": "QUERY",
    "slots": {
        "dishName": "小鸡炖蘑菇"
    }
}
""".strip()


def parse_text(text: str) -> dict:
    resp = client.chat.completions.create(
        model="qwen3:0.6b",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
    )

    content = resp.choices[0].message.content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "text": text,
            "domain": "",
            "intent": "",
            "slots": {},
            "raw_output": content,
            "error": "invalid_json"
        }

    return result


def main():
    input_path = "data/test.json"
    output_path = "data/test_pred.json"

    with open(input_path, "r", encoding="utf-8") as f:
        test_json_list = json.load(f)

    results = []
    for sample in test_json_list:
        text = sample["text"]
        result = parse_text(text)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
    