import random
from fastmcp import FastMCP

mcp = FastMCP(
    name='FastMCP-server',
    instructions='这是一个企业职能助手，可用于查询员工信息、假期信息和会议室状态。'
)

@mcp.tool
def query_department():
    """查询员工所在部门"""
    return random.choice(["技术部", "人事部", "市场部", "财务部"])

@mcp.tool
def query_leave_balance():
    """查询员工剩余年假天数"""
    return f"剩余年假 {random.randint(1, 15)} 天"

@mcp.tool
def query_meeting_room():
    """查询当前可用会议室"""
    return random.choice(["A101会议室可用", "B203会议室可用", "C305会议室占用中"])
