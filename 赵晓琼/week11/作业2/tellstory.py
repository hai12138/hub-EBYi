'''
讲一个故事
'''
import requests
import datetime
from fastmcp import FastMCP

mcp = FastMCP(
    name="tell_story-MCP-Server",
    instructions="This server tells stories",
)

@mcp.tool
def tell_story(title: str = None, genre: str = "童话", protagonist: str = "小猪", length: str = "short"):
    """
    讲故事工具
    参数:
      title: 故事标题（可选）
      genre: 题材，例如 '童话','科幻','悬疑'
      protagonist: 主角名字
      length: 'short'|'medium'|'long'（决定段落数量）
    返回:
      dict: {"story": "...", "meta": {...}}
    """
    try:
        if not title:
            title = f"{protagonist}的{genre}故事"
        para_count = {"short": 2, "medium": 4, "long": 8}.get(length, 2)

        paragraphs = []
        for i in range(para_count):
            paragraphs.append(
                f"第{i+1}段：{protagonist}在一个{genre}的世界中，经历了第{i+1}个冒险，学到了一些重要的东西。"
            )

        story_text = f"{title}\n\n" + "\n\n".join(paragraphs) + "\n\n（完）"
        return {
            "story": story_text,
            "meta": {
                "title": title,
                "genre": genre,
                "protagonist": protagonist,
                "length": length,
                "generated_at": datetime.datetime.utcnow().isoformat(),
            },
        }
    except Exception as e:
        return {"error": str(e)}




