import asyncio
import sqlite3
from typing import Union, Any
import re
import traceback
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
import os
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

set_tracing_disabled(True)
load_dotenv()
external_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_URL"),
)

agent_sql = Agent(
    name="SQLAgent",
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client
    ),
    instructions="""
        你是一个sql处理专家，可以将用户的的提问转换为对应的sql进行回答，
        只回答sql语句，不要有其他多余的回答。
    """,
    handoff_description="""
    获取SQLAgent的SQL结果，并返回给用户。
    """
)

agent_main = Agent(
    name="MainCoordinator",
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client
    ),
    instructions="""
    你是一个智能管家，负责理解用户需求并决定如何处理请求。
    """,
    handoffs=[agent_sql]
)


def extract_code_from_llm(text) -> str:
    pattern = '```sql\n(.*?)```'
    try:
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0]
    except:
        print(traceback.format_exc())
        return ""


async def main(question:str):
    conn = sqlite3.connect('chinook.db')

    cursor = conn.cursor()
    # 获取表名和列名
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    tables_and_columns = cursor.fetchall()
    columns = dict()
    for table_name, table_sql in tables_and_columns:
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        table_columns = cursor.fetchall()
        # print(f"表 {table_name} 的列名: {table_columns}")
        columns[table_name] = table_columns
    instructions = """
        你是一个sql处理专家，可以结合数据库信息将用户的的提问转换为对应的sql进行回答，
        数据库信息为： {#tables_and_columns#}
        请结合上述数据库信息进行sql语句的编写。
        只回答sql语句，不要有其他多余的回答。
        """
    instructions = instructions.replace("{#tables_and_columns#}", str(columns))
    agent_sql.instructions = instructions
    print(agent_sql.instructions)
    # result = await Runner.run(agent_sql, "数据库中总共有多少张表？")
    # result = await Runner.run(agent_sql, "员工表有多少条记录？")
    # result = await Runner.run(agent_sql, "在数据库中所有客户个数和员工个数分别是多少？")
    result = await Runner.run(agent_sql, question)
    # async for event in result.stream_events():
    #     # print( event)
    #     if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
    #         print(event.data.delta, end="", flush=True)
    # print()
    # print(extract_code_from_llm(result.final_output))
    sql = extract_code_from_llm(result.final_output)
    print(sql)
    cursor.execute(sql)
    result = cursor.fetchall()
    print(result)


class SqlAgent:
    def __init__(self):
        self._sql_gen_prompt = """你是一个sql处理专家，可以结合数据库信息将用户的的提问转换为对应的sql进行回答，
        数据库信息为： {#tables_and_columns#}
        请结合上述数据库信息进行sql语句的编写。
        只回答sql语句，不要有其他多余的回答。
        """
        self._code_reflection_prompt = """上述代码执行存在错误，错误信息为{#error#}，请在原有sql基础上进行改进。

        ```sql
        {#code#}
        ```
        """
        self._summary_prompt = """你是答案汇总专家，将用户的提问{#task#} 和 执行结果 {#result#} 汇总为自然语言回答。"""

        self.retry_time = 10

    def get_tables_and_columns(self) -> dict:
        conn = sqlite3.connect('chinook.db')
        cursor = conn.cursor()
        # 获取表名和列名
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        tables_and_columns = cursor.fetchall()
        columns = dict()
        for table_name, table_sql in tables_and_columns:
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            table_columns = cursor.fetchall()
            # print(f"表 {table_name} 的列名: {table_columns}")
            columns[table_name] = table_columns
        return columns

    # 将LLM生成的代码提取出来
    def extract_code_from_llm(self, text) -> str:
        pattern = '```sql\n(.*?)```'
        try:
            matches = re.findall(pattern, text, re.DOTALL)
            return matches[0]
        except:
            print(traceback.format_exc())
            return ""

    # 执行sql
    def execute(self, code, conn) -> Union[bool, Any, str, str]:
        # 超时机制
        try:
            cursor = conn.cursor()
            print('---')
            print(code)
            print('---')
            cursor.execute(code)
            result = cursor.fetchall()  # 执行sql
            return True, result, code, ""
        except Exception as e:
            error_message = traceback.format_exc()
            return False, "", code, error_message

    # 智能体入口
    async def action(self, question) -> str:

        try:
            columns = self.get_tables_and_columns()
            # 生成sql
            instructions_sql = self._sql_gen_prompt.replace("{#tables_and_columns#}", str(columns))
            sql_agent = Agent(
                name="SQLAgent",
                model=OpenAIChatCompletionsModel(
                    model="qwen-max",
                    openai_client=external_client
                ),
                instructions=instructions_sql
            )
            sql_result = await Runner.run(sql_agent, question)
            sql = self.extract_code_from_llm(sql_result.final_output)  # 先抽取代码
            for retry_idx in range(self.retry_time):  # 最多10个尝试
                if sql == "":
                    sql_result = await Runner.run(sql_agent, "the output do not contain any pythnon code using ```sql ```, please generate.")
                    sql = self.extract_code_from_llm(sql_result.final_output)
                else:  # 如果之前抽取了代码 todo
                    conn = sqlite3.connect('chinook.db')
                    execute_issucess, execute_result, code, msg = self.execute(sql, conn)
                    if execute_issucess:
                        result_agent = Agent(
                            name="ResultAgent",
                            model=OpenAIChatCompletionsModel(
                                model="qwen-max",
                                openai_client=external_client
                            )
                        )
                        final_answer = await Runner.run(result_agent, self._summary_prompt.replace("{#task#}", question).replace("{#result#}", str(execute_result)))
                        return final_answer.final_output

                        # 记忆成功提问和代码
                        return code
                    instructions_error = self._code_reflection_prompt.replace("{#error#}", msg)
                    error_agent = Agent(
                        name="ErrorAgent",
                        model=OpenAIChatCompletionsModel(
                            model="qwen-max",
                            openai_client=external_client
                        ),
                        instructions=instructions_error
                    )
                    error_againswer = await Runner.run(error_agent, "请返回错误代码的修改后的代码，请勿返回其他内容")
                    sql = error_againswer.final_output

            print("生成失败")
            return None
        except:
            print(traceback.format_exc())
            return None


if __name__ == "__main__":
    # result = asyncio.run(main(question="数据库中总共有多少张表"))
    # result = asyncio.run(main(question="员工表有多少条记录"))
    # result = asyncio.run(main(question="在数据库中所有客户个数和员工个数分别是多少"))

    # result = asyncio.run(SqlAgent().action(question="数据库中总共有多少张表"))
    # result = asyncio.run(SqlAgent().action(question="员工表有多少条记录"))
    result = asyncio.run(SqlAgent().action(question="在数据库中所有客户个数和员工个数分别是多少"))
    print(result)
