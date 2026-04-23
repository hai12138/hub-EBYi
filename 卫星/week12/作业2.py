# 1. 查询员工请假天数
LEAVE_DATA = {
    "张三": [
        {"date": "2026-04-02", "days": 1, "type": "事假"},
        {"date": "2026-04-10", "days": 2, "type": "病假"},
    ],
    "李四": [
        {"date": "2026-03-15", "days": 3, "type": "年假"},
        {"date": "2026-04-08", "days": 1, "type": "事假"},
    ],
    "王五": [
        {"date": "2026-04-01", "days": 1, "type": "调休"},
    ]
}

@mcp.tool
def query_employee_leave_days(
    employee_name: Annotated[str, "员工姓名"]
):
    """Query the employee's leave days and leave details."""
    records = LEAVE_DATA.get(employee_name, [])
    total_days = sum(item["days"] for item in records)

    return {
        "employee_name": employee_name,
        "total_leave_days": total_days,
        "leave_records": records
    }


# 2. 生成会议纪要
@mcp.tool
def generate_meeting_minutes(
    meeting_topic: Annotated[str, "会议主题"],
    meeting_content: Annotated[str, "会议讨论内容"],
    attendees: Annotated[str, "参会人员，多个姓名可用逗号分隔"] = ""
):
    """Generate structured meeting minutes based on meeting topic and meeting content."""
    attendee_list = [x.strip() for x in attendees.split(",") if x.strip()] if attendees else []

    minutes = f"""会议纪要
一、会议主题
{meeting_topic}

二、参会人员
{', '.join(attendee_list) if attendee_list else '未提供'}

三、会议内容摘要
{meeting_content}

四、待办事项
1. 根据会议讨论内容推进相关工作。
2. 明确责任人和完成时间。
3. 后续跟进执行结果并同步进展。
"""

    return {
        "meeting_topic": meeting_topic,
        "attendees": attendee_list,
        "meeting_minutes": minutes
    }


# 3. 计算报销金额
@mcp.tool
def calculate_reimbursement_amount(
    transport_fee: Annotated[Union[int, float], "交通费"],
    hotel_fee: Annotated[Union[int, float], "住宿费"],
    meal_fee: Annotated[Union[int, float], "餐饮费"],
    other_fee: Annotated[Union[int, float], "其他费用"] = 0
):
    """Calculate the total reimbursement amount."""
    total = float(transport_fee) + float(hotel_fee) + float(meal_fee) + float(other_fee)

    return {
        "transport_fee": float(transport_fee),
        "hotel_fee": float(hotel_fee),
        "meal_fee": float(meal_fee),
        "other_fee": float(other_fee),
        "total_amount": round(total, 2)
    }
