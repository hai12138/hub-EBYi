import re
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

EMPLOYEE_PROFILE = {
    "张三": {
        "department": "研发中心",
        "title": "后端工程师",
        "email": "zhangsan@company.local",
        "extension": "6812",
        "office": "深圳南山 A 座 5F",
    },
    "李四": {
        "department": "人力资源部",
        "title": "HRBP",
        "email": "lisi@company.local",
        "extension": "5201",
        "office": "深圳南山 B 座 3F",
    },
    "王五": {
        "department": "财务部",
        "title": "薪酬专员",
        "email": "wangwu@company.local",
        "extension": "4038",
        "office": "深圳福田 C 座 8F",
    },
}

LEAVE_BALANCE = {
    "张三": {"annual_total": 15, "used": 5, "remaining": 10, "carry_over": 2, "last_updated": "2026-04-10"},
    "李四": {"annual_total": 12, "used": 4, "remaining": 8, "carry_over": 1, "last_updated": "2026-04-09"},
    "王五": {"annual_total": 20, "used": 11, "remaining": 9, "carry_over": 0, "last_updated": "2026-04-08"},
}

PAYROLL_INFO = {
    "张三": {"last_pay_date": "2026-03-28", "next_pay_date": "2026-04-28", "status": "已进入银行代发", "bank_tail_no": "6228"},
    "李四": {"last_pay_date": "2026-03-28", "next_pay_date": "2026-04-28", "status": "薪资单已生成", "bank_tail_no": "1024"},
    "王五": {"last_pay_date": "2026-03-28", "next_pay_date": "2026-04-28", "status": "待财务复核", "bank_tail_no": "8876"},
}

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)


def _lookup_user_record(user_name: str, source_data: dict, record_name: str):
    user_name = user_name.strip()
    if user_name in source_data:
        return {"user_name": user_name, **source_data[user_name]}

    return {
        "user_name": user_name,
        "error": f"未查询到 {record_name} 信息，请使用 张三 / 李四 / 王五 进行演示。",
    }

@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []

@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []

@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []

@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []

@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."], 
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."], 
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    positive_keywords_zh = ['喜欢', '赞', '棒', '优秀', '精彩', '完美', '开心', '满意']
    negative_keywords_zh = ['差', '烂', '坏', '糟糕', '失望', '垃圾', '厌恶', '敷衍']

    positive_pattern = '(' + '|'.join(positive_keywords_zh) + ')'
    negative_pattern = '(' + '|'.join(negative_keywords_zh) + ')'

    positive_matches = re.findall(positive_pattern, text)
    negative_matches = re.findall(negative_pattern, text)

    count_positive = len(positive_matches)
    count_negative = len(negative_matches)

    if count_positive > count_negative:
        return "积极 (Positive)"
    elif count_negative > count_positive:
        return "消极 (Negative)"
    else:
        return "中性 (Neutral)"


@mcp.tool
def query_salary_info(user_name: Annotated[str, "用户名"]):
    """Query user salary baed on the username."""

    # TODO 基于用户名，在数据库中查询，返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000


@mcp.tool
def query_employee_leave_balance(user_name: Annotated[str, "员工姓名，例如：张三"]):
    """查询员工年假信息，包括总天数、已用天数、剩余天数和最近更新时间。"""
    return _lookup_user_record(user_name, LEAVE_BALANCE, "年假")


@mcp.tool
def query_salary_payment_schedule(user_name: Annotated[str, "员工姓名，例如：张三"]):
    """查询员工最近一次发薪日期、下一次发薪日期、发放状态以及银行卡尾号。"""
    return _lookup_user_record(user_name, PAYROLL_INFO, "薪资发放")


@mcp.tool
def query_employee_contact_card(user_name: Annotated[str, "员工姓名，例如：张三"]):
    """查询员工通讯录信息，包括部门、岗位、邮箱、分机号和办公地点。"""
    return _lookup_user_record(user_name, EMPLOYEE_PROFILE, "员工通讯录")

