import json
from openai import OpenAI
client = OpenAI(
    api_key='<KEY>',
    base_url='http://localhost:11434/v1/'
)


def chat(text: str) -> str:

    resp = client.chat.completions.create(
        model='qwen3:0.6b',
        messages=[
            {'role': 'system', 'content': '''
            你是意图分类和命名实体识别专家，请对给出的文本进行意图分类和命名实体识别，只返回类似以下格式的json对象
            不返回其他的非json内容
            text为输入文本
            domain为输入文本的领域 领域的待选标签 music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
            intent为输入文本意图 意图的待选标签 OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
            slots为输入文本提取的实体 实体的内容是在文本中抽取的 实体的待选标签 code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time
{
        "text": "小鸡炖蘑菇的做法爸", 
        "domain": "cookbook", 
        "intent": "QUERY", 
        "slots": {
            "dishName": "小鸡炖蘑菇"
        }
    }'''},
            {"role": "user", "content": text},
        ]
    )
    # print(resp)
    if resp and resp.choices[0]:
        print(resp.choices[0].message.content)
    else:
        print('error')

# chat('test_json')
with open('data/test.json', 'r', encoding='utf-8') as f:
    test_json_list = json.load(f)
    for i,test_json in enumerate(test_json_list):
        if i > 1:
            break
        # print(i)
        # print(test_json)
        chat(test_json['text'])
        print('\n\n')


