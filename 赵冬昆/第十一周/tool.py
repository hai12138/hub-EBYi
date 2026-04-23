from fastmcp import FastMCP
import datetime
import requests

import re
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

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


# === 工具1：获取当前时间与节日提醒 ===
HOLIDAYS = {
    "01-01": "元旦",
    "02-14": "情人节",
    "03-08": "妇女节",
    "05-01": "劳动节",
    "05-04": "青年节",
    "06-01": "儿童节",
    "09-10": "教师节",
    "10-01": "国庆节",
    "12-25": "圣诞节",
}


@mcp.tool()
def get_current_time_and_holiday():
    """获取当前日期、星期和节日信息"""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y年%m月%d日")
    weekday = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
    today_mmdd = now.strftime("%m-%d")
    holiday = HOLIDAYS.get(today_mmdd, "无特殊节日")
    return {
        "date": date_str,
        "weekday": f"星期{weekday}",
        "holiday": holiday
    }


# === 工具2（新）：查询A股股票实时行情 ===
@mcp.tool()
def get_stock_price(symbol: str):
    """
    查询沪深A股实时行情（输入6位股票代码，如 '600519'）
    数据来源: 麦睿免费API（无需key）
    """
    if not symbol or not isinstance(symbol, str):
        return {"error": "请输入6位股票代码，如 '600519'"}

    # 清理输入（只保留数字）
    clean_symbol = ''.join(filter(str.isdigit, symbol))
    if len(clean_symbol) != 6:
        return {"error": "股票代码必须是6位数字"}

    try:
        # 使用公开免费接口（已验证可用）
        url = f"https://api.mairui.club/hsrl/ssjy/{clean_symbol}/55798CC8-F964-4E44-8156-410750E40146"
        # https://api.mairuiapi.com/hslt/list/您的licence
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get("code") == 200 and data.get("data"):
            d = data["data"]
            return {
                "stock_name": d.get("name", "未知"),
                "symbol": clean_symbol,
                "current_price": d.get("dqj", "N/A"),
                "change_percent": d.get("zf", "N/A") + "%",
                "high": d.get("zgj", "N/A"),
                "low": d.get("zdj", "N/A"),
                "volume": d.get("cjl", "N/A"),  # 成交量（手）
                "amount": d.get("cje", "N/A")  # 成交额（万元）
            }
        else:
            return {"error": "未找到该股票或已停牌"}
    except Exception as e:
        return {"error": f"查询失败: {str(e)}"}


# === 工具3：BMI 肥胖判断工具 ===
@mcp.tool()
def check_obesity(height_cm: float, weight_kg: float):
    """根据身高（厘米）和体重（公斤）计算 BMI 并判断是否肥胖"""
    if height_cm <= 0 or weight_kg <= 0:
        return {"error": "身高和体重必须为正数"}

    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2)

    if bmi < 18.5:
        category = "偏瘦"
    elif bmi < 24:
        category = "正常"
    elif bmi < 28:
        category = "超重"
    else:
        category = "肥胖"

    return {
        "bmi": bmi,
        "category": category,
        "advice": "建议保持健康饮食和规律运动" if category in ["超重", "肥胖"] else "保持良好状态！"
    }
