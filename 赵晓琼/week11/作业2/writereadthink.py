'''
写读后感
'''
import requests
import datetime
from typing import Optional
from fastmcp import FastMCP

mcp = FastMCP(
    name="write-read-think-MCP-Server",
    instructions="This server generates short reading reflections (读后感).",
)

@mcp.tool(description="生成读后感：支持 title/author/length/tone/max_chars 参数，返回结构化结果")
def write_read_think(
    title: Optional[str] = None,
    author: str = "莫言",
    length: str = "short",   # one of "short","medium","long"
    tone: str = "平实",      # e.g. "平实","感慨","严肃","温暖"
    max_chars: int = 100     # 强制最大字符数
):
    """
    讲读后感工具
    参数:
      title: 书名或文章标题（可选）
      author: 作者名，默认 "莫言"
      length: 输出长度档位："short" (约100字) / "medium" (约200字) / "long" (约400字)
      tone: 语气风格，例如 "平实","感慨","温暖"
      max_chars: 强制的最大字符数上限（最终结果不会超过此长度）
    返回:
      dict: {
        "content": "...",           # 读后感正文（字符串）
        "meta": {...}               # 生成元信息
      }
    """
    if not title:
        title = "《父亲的皮带》"

    # 估算字符上限（以 length 档位为基础，再受 max_chars 约束）
    length_map = {"short": 100, "medium": 200, "long": 400}
    target_chars = length_map.get(length, 100)
    limit = min(target_chars, max_chars)

    # 生成模板化内容（可替换为更复杂的生成逻辑或调用写作模型）
    try:
        base_sentences = []

        # 开头：点明读物与第一印象
        base_sentences.append(f"读完《{title}》，作者 {author} 的笔触让我很受触动。")

        # 中段：根据语气加入不同描述
        if tone in ("感慨", "感伤"):
            base_sentences.append("文字里有时代的重量，让人回想起那些不易被忘记的细节与情感。")
            base_sentences.append("每个细微的场景都像一面镜子，映出人性的复杂与温柔。")
        elif tone in ("温暖",):
            base_sentences.append("叙述中流露出对人性的理解与温情，使人读后心头一暖。")
            base_sentences.append("细腻的描写让人物更立体，情感更真切。")
        elif tone in ("严肃",):
            base_sentences.append("作品以克制的笔法拆解了社会与人性的难题，读来沉重但有力量。")
            base_sentences.append("文本强调事实与记忆，留给读者持续的思考空间。")
        else:  # 平实或默认
            base_sentences.append("文笔朴实却富有力量，叙事节奏稳健，细节处理到位。")
            base_sentences.append("短短几段文字勾勒出鲜明的情绪与场景。")

        # 结尾：感悟或建议
        base_sentences.append("总体而言，这是一篇值得反复回味的作品，让人既感动又沉思。")
        base_sentences.append("推荐给喜欢细腻叙事与沉淀情感的读者。")

        # 拼接并裁剪到字符限制
        content = " ".join(base_sentences).strip()
        if len(content) > limit:
            content = content[:limit].rstrip()
            # 避免截断到半个标点，统一以省略号结尾
            if not content.endswith(("。", "！", "？")):
                content = content.rstrip("，,") + "..."
        meta = {
            "title": title,
            "author": author,
            "length_requested": length,
            "tone": tone,
            "max_chars": max_chars,
            "generated_at": datetime.datetime.utcnow().isoformat(),
        }

        return {"content": content, "meta": meta}
    except Exception as e:
        return {"error": str(e)}

