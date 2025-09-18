from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List


# 1. 定义状态结构：存储对话信息和追问状态
class BookingState(TypedDict):
    user_input: str  # 用户当前输入
    history: List[str]  # 对话历史
    checkin_date: Optional[str]  # 入住日期（需追问的信息1）
    room_type: Optional[str]  # 房型（需追问的信息2）
    need_followup: bool  # 是否需要追问
    followup_question: Optional[str]  # 追问问题


# 2. 定义节点函数

def check_information(state: BookingState) -> BookingState:
    """判断节点：检查是否有缺失的必要信息，决定是否需要追问"""
    # 提取当前用户输入中的信息（实际场景可结合LLM解析，这里简化为字符串匹配）
    user_input = state["user_input"].lower()
    new_state = state.copy()

    # 提取入住日期（假设用户输入含"xx月xx日"则视为已提供）
    if "月" in user_input and "日" in user_input:
        new_state["checkin_date"] = user_input.split("入住")[1].strip()  # 简化提取
    # 提取房型（假设含"大床房"/"双床房"则视为已提供）
    if "大床" in user_input:
        new_state["room_type"] = "大床房"
    elif "双床" in user_input:
        new_state["room_type"] = "双床房"

    # 检查是否有缺失信息，生成追问问题
    missing = []
    if not new_state["checkin_date"]:
        missing.append("入住日期（如：10月1日）")
    if not new_state["room_type"]:
        missing.append("房型（大床房/双床房）")

    if missing:
        # 需要追问：更新状态
        new_state["need_followup"] = True
        new_state["followup_question"] = f"请补充以下信息：{'; '.join(missing)}"
    else:
        # 信息完整：无需追问，进入处理流程
        new_state["need_followup"] = False
        new_state["followup_question"] = None

    # 更新对话历史
    new_state["history"] = state["history"] + [f"用户：{state['user_input']}"]
    return new_state


def generate_followup(state: BookingState) -> BookingState:
    """追问节点：返回追问问题给用户"""
    new_state = state.copy()
    # 将追问问题加入对话历史
    new_state["history"].append(f"系统：{state['followup_question']}")
    return new_state


def process_booking(state: BookingState) -> BookingState:
    """处理节点：信息完整时，执行预订逻辑"""
    new_state = state.copy()
    # 生成预订结果（实际场景可调用酒店API）
    result = (
        f"已为您预订：{state['room_type']}，入住日期：{state['checkin_date']}。\n"
        "订单号：HS20241001001，凭身份证办理入住。"
    )
    new_state["history"].append(f"系统：{result}")
    return new_state


# 3. 构建状态图
graph = StateGraph(BookingState)

# 添加节点
graph.add_node("check_info", check_information)  # 判断是否需要追问
graph.add_node("followup", generate_followup)    # 生成追问
graph.add_node("book", process_booking)          # 处理预订

# 4. 定义边（控制流程走向）
# 入口 → 判断节点
graph.set_entry_point("check_info")

# 判断节点 → 分支：需要追问则到追问节点，否则到处理节点
def should_followup(state: BookingState) -> str:
    return "followup" if state["need_followup"] else "book"

graph.add_conditional_edges(
    source="check_info",
    condition=should_followup,
    # 条件映射：返回值 → 目标节点
    mapping={
        "followup": "followup",
        "book": "book"
    }
)

# 追问节点 → 等待用户输入后，回到判断节点（形成循环）
graph.add_edge("followup", "check_info")

# 处理节点 → 结束
graph.add_edge("book", END)

# 编译图
compiled_graph = graph.compile()


# 5. 模拟对话流程
def run_dialog():
    # 初始状态
    current_state = {
        "user_input": "",
        "history": [],
        "checkin_date": None,
        "room_type": None,
        "need_followup": False,
        "followup_question": None
    }

    print("开始酒店预订（输入'退出'结束）\n")
    while True:
        # 获取用户输入
        user_input = input("用户：")
        if user_input.lower() == "退出":
            break
        current_state["user_input"] = user_input

        # 执行图
        current_state = compiled_graph.invoke(current_state)

        # 输出系统回复（最后一条历史）
        print(f"系统：{current_state['history'][-1].split('系统：')[-1]}\n")

        # 如果已完成预订，结束对话
        if "已为您预订" in current_state["history"][-1]:
            break


if __name__ == "__main__":
    run_dialog()
