import os
import logging
from agents import Agent, Runner, RunConfig, ModelSettings, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent, ResponseCreatedEvent, ResponseOutputItemDoneEvent, \
    ResponseFunctionToolCall
from agents import set_default_openai_api, set_tracing_disabled

from sub_agents.sentiment_agent import sentiment_agent
from sub_agents.entity_agent import entity_agent

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

logger = logging.getLogger(__name__)

external_client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

GATE_PROMPT = (
    "你是一个智能体路由器，你的任务是根据用户的问题，选择合适的智能体来处理。"
)

class TaskManager:
    async def run(self, query: str):
        gate_agent = Agent(
            name="gate_agent",
            instructions=GATE_PROMPT,
            handoffs=[sentiment_agent, entity_agent],
            model=OpenAIChatCompletionsModel(
                model="qwen-max",
                openai_client=external_client,
            ),
            model_settings=ModelSettings(parallel_tool_calls=False)
        )

        result = Runner.run_streamed(
            gate_agent,
            input=query,
            run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False))
        )

        print("\n=== 开始响应 ===\n")

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                await self._handle_raw_response_event(event)
            elif event.type == "run_item_stream_event":
                await self._handle_run_item_stream_event(event)
            elif event.type == "response_completed_event":
                print("\n=== 响应完成 ===")
            elif event.type == "error_event":
                print(f"\n[错误] {event}")

    async def _handle_raw_response_event(self, event):
        if hasattr(event, 'data'):
            data = event.data

            if isinstance(data, ResponseTextDeltaEvent):
                print(data.delta, end='', flush=True)

            elif isinstance(data, ResponseOutputItemDoneEvent):
                if isinstance(data.item, ResponseFunctionToolCall):
                    func_name = data.item.name
                    func_args = data.item.arguments
                    print(f"\n\n[函数调用] {func_name}", flush=True)
                    print(f"[参数] {func_args}", flush=True)

    async def _handle_run_item_stream_event(self, event):
        if hasattr(event, 'name'):
            if event.name == "tool_output":
                output = event.item.raw_item.get("output") if hasattr(event.item, 'raw_item') else None
                print(f"\n[工具输出] {output}", flush=True)
            elif event.name == "tool_use":
                tool_name = event.item.raw_item.get("name") if hasattr(event.item, 'raw_item') else None
                print(f"\n[使用工具] {tool_name}", flush=True)
