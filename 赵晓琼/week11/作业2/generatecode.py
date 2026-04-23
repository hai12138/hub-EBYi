# ...existing code...
import datetime
from fastmcp import FastMCP

mcp = FastMCP(
    name="generate_code-MCP-Server",
    instructions="This server generate code",
)

@mcp.tool
def generate_code(title: str = None, language: str = "python"):
    """
    生成简单算法代码示例
    参数:
      title: 要生成的算法描述，例如 "二叉树中序遍历"、"冒泡排序"、"快速排序"、"斐波那契"
      language: 目标语言，可选 "python","java","cpp"
    返回:
      dict: {
        "code": "...",
        "language": "...",
        "description": "...",
        "meta": {...}
      }
    """
    try:
        algo = (title or "").strip().lower()
        if not algo:
            algo = "二叉树中序遍历"

        lang = (language or "python").lower()
        def py_bst_inorder():
            return """class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder(root):
    res = []
    def dfs(node):
        if not node: return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)
    dfs(root)
    return res
"""

        def py_bubble_sort():
            return """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""

        def py_quick_sort():
            return """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)
"""

        def py_fibonacci():
            return """def fib(n):
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a + b
    return b
"""

        def cpp_bubble_sort():
            return """#include <vector>
using namespace std;
vector<int> bubble_sort(vector<int>& arr){
    int n = arr.size();
    for(int i=0;i<n;i++){
        for(int j=0;j<n-i-1;j++){
            if(arr[j] > arr[j+1]) swap(arr[j], arr[j+1]);
        }
    }
    return arr;
}
"""

        # 简单匹配关键词
        if "中序" in algo or "二叉树" in algo:
            snippet = {
                "python": py_bst_inorder(),
                "java": "// Java 中序遍历示例：请按需实现 TreeNode 和 递归方法",
                "cpp": "// C++ 中序遍历示例：请按需实现 TreeNode 与递归函数"
            }.get(lang, py_bst_inorder())
            desc = "二叉树中序遍历示例"
        elif "冒泡" in algo or "bubble" in algo:
            snippet = {
                "python": py_bubble_sort(),
                "java": "// Java 冒泡排序示例：实现数组排序逻辑",
                "cpp": cpp_bubble_sort()
            }.get(lang, py_bubble_sort())
            desc = "冒泡排序"
        elif "快" in algo or "快速" in algo or "quick" in algo:
            snippet = {
                "python": py_quick_sort(),
                "java": "// Java 快速排序示例：实现分区与递归",
                "cpp": "// C++ 快速排序示例：实现分区与递归"
            }.get(lang, py_quick_sort())
            desc = "快速排序"
        elif "斐波" in algo or "fibo" in algo:
            snippet = {
                "python": py_fibonacci(),
                "java": "// Java 斐波那契示例：迭代或递归实现",
                "cpp": "// C++ 斐波那契示例：迭代或递归实现"
            }.get(lang, py_fibonacci())
            desc = "斐波那契数列"
        else:
            # 默认：返回一个 Python 模板，说明如何实现
            snippet = {
                "python": py_quick_sort()
            }.get(lang, py_quick_sort())
            desc = f"未识别算法，返回示例 ({algo})"

        return {
            "code": snippet,
            "language": lang,
            "description": desc,
            "meta": {
                "requested_title": title,
                "generated_at": datetime.datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        return {"error": str(e)}