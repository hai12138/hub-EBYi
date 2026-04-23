import re
from typing import Annotated, Union
import requests
TOKEN = "738b541a5f7a"

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

# # 1. 查找今天体育新闻
# @mcp.tool
# def get_today_sports_news(query: Annotated[str, "体育相关关键词，可不填"] = ""):
#     """获取今天最新的体育新闻，支持关键词筛选。"""
#     try:
#         url = f"https://whyta.cn/api/tx/tiyuxinwen?msg={query}"
#         return requests.get(url).json().get("news", [])
#     except:
#         return []
#
# # 2. 查找今天科技新闻
# @mcp.tool
# def get_today_tech_news(query: Annotated[str, "科技相关关键词，可不填"] = ""):
#     """获取今天最新的科技新闻，支持关键词筛选。"""
#     try:
#         url = f"https://whyta.cn/api/tx/kejixinwen?msg={query}"
#         return requests.get(url).json().get("news", [])
#     except:
#         return []
#
# # 3. 查找今天历史上的今天事件
# @mcp.tool
# def get_today_history_events():
#     """获取历史上的今天发生的重要事件。"""
#     try:
#         url = "https://whyta.cn/api/tx/lishishangdejintian"
#         return requests.get(url).json().get("result", [])
#     except:
#         return []
# 1. 查找今天体育新闻
@mcp.tool
def get_today_sports_news(news_keyword: Annotated[str, "Sports news keyword"]):
    """Retrieves today's sports news based on the provided keyword."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/tiyuxinwen?key={TOKEN}&word={news_keyword}").json()["result"]
    except:
        return []

# 2. 查找今天科技新闻
@mcp.tool
def get_today_tech_news(news_keyword: Annotated[str, "Tech news keyword"]):
    """Retrieves today's technology news based on the provided keyword."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/kejixinwen?key={TOKEN}&word={news_keyword}").json()["result"]
    except:
        return []

# 3. 查找今天历史事件（历史上的今天）
@mcp.tool
def get_today_history_events(date_info: Annotated[str, "Date for querying historical events"]):
    """Retrieves historical events that occurred on this day in history."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/lishishangdejintian?key={TOKEN}&word={date_info}").json()["result"]
    except:
        return []