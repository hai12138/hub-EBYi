"""
作业2:
4-项目案例-企业职能助手，增加3个自定义的tool 工具，实现自定义的功能，并在对话框完成调用（自然语言 -》 工具选择 -》 工具执行结果）
"""
from fastmcp import FastMCP
from typing import Annotated
mcp = FastMCP(
    name="Company-MCP-Server",
    instructions="""这个MCP服务器提供了几个工具，帮助员工查询公司相关信息，包括员工信息、节假日安排和最新公告。请根据用户的自然语言请求，选择合适的工具进行查询，并返回结果。""",
)


EMPLOYEE_DB = {
    "EMP100001": {"name": "张三", "department": "研发部", "entry_date": "2021-03-15", "position": "工程师"},
    "EMP100002": {"name": "李四", "department": "市场部", "entry_date": "2020-07-01", "position": "市场经理"},
    "EMP100003": {"name": "王五", "department": "人事部", "entry_date": "2019-11-20", "position": "HR"},
}

@mcp.tool
def get_employee_info(ID: Annotated[str, "员工工号，如 EMP100001"]):
    """Display employee information (such as name, department, entry date, position, etc.) based on the provided employee ID."""
    return EMPLOYEE_DB.get(ID, "未找到该员工信息")



@mcp.tool
def get_company_holidays(
    year: Annotated[int, "年份，如 2026"]
):
    """Query company holiday arrangements (e.g., New Year's Day, Spring Festival, National Day, etc.)"""

    holidays = {
        2026: [
            {"name": "元旦", "start": "2026-01-01", "end": "2026-01-03"},
            {"name": "春节", "start": "2026-02-17", "end": "2026-02-23"},
            {"name": "国庆", "start": "2026-10-01", "end": "2026-10-07"},
        ]
    }
    return holidays.get(year, "未找到该年份的节假日安排")

@mcp.tool
def get_latest_announcements():
    """Query the latest company announcements (e.g., meeting notices, policy updates, etc.)"""
    announcements = [
        {"date": "2026-04-10", "title": "关于五一放假安排的通知"},
        {"date": "2026-03-28", "title": "2026年度体检通知"},
        {"date": "2026-03-01", "title": "新员工入职培训安排"},
    ]
    return announcements
