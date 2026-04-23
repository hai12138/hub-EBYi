4-项目案例-企业职能助手，增加3个自定义的tool 工具，实现自定义的功能，并在对话框完成调用（自然语言 -》 工具选择 -》 工具执行结果）
'''
import random

from fastmcp import FastMCP
mcp = FastMCP(
    name='FastMCP-server',
    instructions='',
)
@mcp.tool
def query_age():
    '''查询年龄'''
    return random.randint(20,30)

@mcp.tool
def is_married():
    '''查询是否已婚'''
    if random.random() > 0.5:
        return '是'
    return '否'
@mcp.tool
def query_job():
    '''查询做什么工作'''
    if random.random() > 0.5:
        return '老师'
    return '工人'
