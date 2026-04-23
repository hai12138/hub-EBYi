import re
from typing import Annotated, Union, List, Dict
import requests
API_KEY = "sk-ba21cb2ae67542f194bd29851f5db6c3"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Educations-MCP-Server",
    instructions="""This server contains some api of educations.""",
)

@mcp.tool
def get_brain_teaser(nums:Annotated[int, "脑筋急转弯的数量"]):
    """ """
    try:
        return requests.get(f"https://whyta.cn/api/tx/naowan?key={API_KEY}&num={nums}").json()["result"]["list"]
    except Exception as e:
        return []

@mcp.tool
def get_classic_lines():
    """ """
    try:
        return requests.get(f"https://whyta.cn/api/tx/dialogue?key={API_KEY}").json()["result"]
    except Exception as e:
        return []

@mcp.tool
def get_gold_price():
    """ """
    try:
        return requests.get(f"https://whyta.cn/api/goldprice?key={API_KEY}").json()["result"]["data"]
    except Exception as e:
        return []

@mcp.tool
def entity_recognition(text:Annotated[str, "待进行实体识别的文本 / The text for entity recognition"]):
    """Extract and recognize named entities (person, location, organization......) from the input text."""
    entities:Dict[str, List[str]] = {
        "人名":[],
        "地名":[],
        "机构名":[],
        "时间":[],
    }

    # 1. 人名
    name_pattern = re.compile(
        r"[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方"
        r"俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邵常万"
        r"][\u4e00-\u9fa5]{1,2}"

    )
    persons = name_pattern.findall(text)
    entities["人名"] = list(set(persons))

    # 2. 地名
    loc_pattern = re.compile(
        r"([北京|天津|上海|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|广西|海南|四川|贵州|云南|西藏|陕西|甘肃|青海|宁夏|新疆|"
        r"济南|青岛|淄博|枣庄|东营|烟台|潍坊|济宁|泰安|威海|日照|临沂|德州|聊城|滨州|菏泽|"
        r"[省市县区镇村街路公园广场医院学校大厦小区港口机场车站]+)"
    )
    losc = loc_pattern.findall(text)
    entities["地名"] = list(set(losc))

    # 3. 机构名
    org_pattern = re.compile(r"([^\s]{2,}(公司|集团|大学|中学|小学|医院|银行|局|厅|中心|出版社))")
    orgs = org_pattern.findall(text)
    entities["机构名"] = list(set([o[0] for o in orgs]))

    # 4. 时间
    time_pattern = re.compile(
        r"(\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日|星期[一二三四五六日天]|春天|夏天|秋天|冬天)")
    times = time_pattern.findall(text)
    entities["时间"] = list(set(times))

    # ==========================
    # 输出格式化结果
    # ==========================
    output = "【专业实体识别结果】\n"
    for k, v in entities.items():
        if v:
            output += f"{k}：{'、'.join(v)}\n"

    if all(len(v) == 0 for v in entities.values()):
        output += "未识别到任何实体"

    return output
