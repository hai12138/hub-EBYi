from agents import tool

# Tool 1：计算员工加班时长
@tool
def calculate_overtime_hours(work_hours: float, standard_hours: float = 8) -> float:
    """
    计算员工当日加班时长
    work_hours: 实际工作小时
    standard_hours: 标准工时，默认8小时
    返回：加班小时数
    """
    overtime = max(0, work_hours - standard_hours)
    return overtime

# Tool 2：查询部门员工人数（模拟）
@tool
def get_department_employee_count(department_name: str) -> int:
    """
    查询某部门员工数量（模拟数据）
    department_name: 部门名称
    返回：人数
    """
    dept_count = {
        "技术部": 25,
        "人事部": 8,
        "财务部": 6,
        "市场部": 15
    }
    return dept_count.get(department_name, 0)

# Tool 3：生成企业通知摘要
@tool
def generate_company_notice_summary(notice_content: str) -> str:
    """
    对企业通知进行精简摘要
    notice_content: 原始通知文本
    返回：3句话以内摘要
    """
    return f"【通知摘要】\n{notice_content[:50]}..."
