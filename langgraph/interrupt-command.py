from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Literal
import datetime
import re
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
import uuid

# 1. 状态定义
class OrderState(TypedDict):
    user_input: str
    history: List[str]
    intent: Literal["query_order", "refund_application", "unknown"]
    order_id: Optional[str]
    refund_reason: Optional[str]
    need_followup: bool
    followup_question: Optional[str]
    tool_result: Optional[str]
    followup_count: int
    current_node: str

def query_order(order_id: str) -> str:
    print("query_order", order_id)
    if order_id.startswith("ORD") and len(order_id) == 10:
        return (f"订单 {order_id} 状态：已发货\n"
                f"物流信息: xxxx\n"
                f"预计送达时间：{datetime.date.today() + datetime.timedelta(days=2)}")
    else:
        return f"⚠️  订单查询失败：订单号「{order_id}」格式无效（需10位，以ORD开头，如ORD202400001）"

def apply_refund(order_id: str, reason: str) -> str:
    if order_id.startswith("ORD") and len(order_id) == 10:
        return (f"✅ 退款申请已受理\n"
                f"订单号：{order_id}\n"
                f"退款原因：{reason}\n"
                f"预计处理时间：1-3个工作日")
    else:
        return f"⚠️  退款申请失败：订单号「{order_id}」格式无效（需10位，以ORD开头，如ORD202400001）"

# 3. 节点函数
def detect_intent(state: OrderState) -> OrderState:
    new_state = state.copy()
    user_input = new_state["user_input"].lower().strip()
    print('user_input', user_input)
    new_state["need_followup"] = False
    new_state["followup_question"] = None
    if any(keyword in user_input for keyword in ["查", "查询", "状态", "物流", "订单", "ORD"]):
        new_state["intent"] = "query_order"
    elif any(keyword in user_input for keyword in ["退", "退款", "退货", "取消"]):
        new_state["intent"] = "refund_application"
    else:
        new_state["intent"] = "unknown"
        new_state["need_followup"] = True
        new_state["followup_question"] = "抱歉，我没理解您的需求～请问您需要「查询订单」还是「申请退款」呢？"
    new_state["current_node"] = "detect_intent"
    print(f"\n[状态快照] {new_state['current_node']} -> 意图：{new_state['intent']}，需追问：{new_state['need_followup']}")
    return new_state

def check_information(state: OrderState) -> OrderState:
    new_state = state.copy()
    user_input = new_state["user_input"].strip()
    intent = new_state["intent"]
    max_followup = 3

    if new_state["followup_count"] >= max_followup:
        new_state["tool_result"] = f"⚠️  已尝试追问{max_followup}次仍未获取有效信息，建议您重新描述需求（如「查询订单 ORD202400001」）"
        new_state["current_node"] = "check_information"
        print(f"\n[状态快照] {new_state['current_node']} -> 追问次数超限，终止追问")
        # 重置标记
        return Command(goto="detect_intent")
    # 重置标记
    new_state["need_followup"] = False
    new_state["followup_question"] = None

    if not new_state["order_id"]:
        order_match = re.search(r"ORD\d{7}", user_input)
        if order_match:
            new_state["order_id"] = order_match.group()
            print(f"[信息提取] 订单号：{new_state['order_id']}")
        else:
            new_state["need_followup"] = True
            new_state["followup_count"] += 1
            new_state["followup_question"] = f"（{new_state['followup_count']}/{max_followup}）请提供10位订单号（以ORD开头，如ORD202400001）"
            new_state["current_node"] = "check_information"
            print(f"\n[状态快照] {new_state['current_node']} -> 缺订单号，追问次数：{new_state['followup_count']}")
            return new_state

    if intent == "refund_application" and not new_state["refund_reason"]:
        reason = re.sub(r"ORD\d{7}", "", user_input).strip()
        if reason and reason not in ["无", "空", "不知道"]:
            new_state["refund_reason"] = reason
            print(f"[信息提取] 退款原因：{new_state['refund_reason']}")
        else:
            new_state["need_followup"] = True
            new_state["followup_count"] += 1
            new_state["followup_question"] = f"（{new_state['followup_count']}/{max_followup}）请说明退款原因（如「质量问题」「尺寸不符」）"
            new_state["current_node"] = "check_information"
            print(f"\n[状态快照] {new_state['current_node']} -> 缺退款原因，追问次数：{new_state['followup_count']}")
            return new_state

    new_state["current_node"] = "check_information"
    print(f"\n[状态快照] {new_state['current_node']} -> 信息齐全")
    return new_state

def generate_followup(state: OrderState) -> OrderState:
    new_state = state.copy()
    if new_state["followup_question"]:
        new_state["history"].append(f"系统：{new_state['followup_question']}")
    new_state["current_node"] = "followup"
    print(f"\n[状态快照] {new_state['current_node']} -> 追问内容：{new_state['followup_question']}")
    return interrupt(state)


def call_tool(state: OrderState) -> OrderState:
    new_state = state.copy()
    intent = new_state["intent"]
    order_id = new_state["order_id"]

    if intent == "query_order":
        result = query_order(order_id)
    elif intent == "refund_application":
        result = apply_refund(order_id, new_state["refund_reason"])
    else:
        result = "⚠️  无法识别您的需求，请重新输入"
    
    new_state["tool_result"] = result
    new_state["history"].append(f"系统：{result}")
    new_state["current_node"] = "call_tool"
    print(f"\n[状态快照] {new_state['current_node']} -> 工具调用完成")
    return new_state

# 4. 状态图构建
def create_order_graph():
    graph = StateGraph(OrderState)
    
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("check_information", check_information)
    graph.add_node("followup", generate_followup)
    graph.add_node("call_tool", call_tool)
    
    graph.set_entry_point("detect_intent")

    # 意图识别后分支
    def after_intent(state: OrderState) -> str:
        return "followup" if state["need_followup"] else "check_information"
    graph.add_conditional_edges("detect_intent", after_intent, {"followup": "followup", "check_information": "check_information"})

    # 信息检查后分支
    def after_check(state: OrderState) -> str:
        return "followup" if state["need_followup"] else "call_tool"
    graph.add_conditional_edges("check_information", after_check, {"followup": "followup", "call_tool": "call_tool"})

    graph.add_edge("followup", "check_information")
    graph.add_edge("call_tool", END)
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)

# 5. 修复后的对话逻辑
def run_conversation():
    # 初始化状态
    current_state = OrderState(
        user_input="",
        history=[],
        intent="unknown",
        order_id=None,
        refund_reason=None,
        need_followup=False,
        followup_question=None,
        tool_result=None,
        followup_count=0,
        current_node=""
    )
    
    app = create_order_graph()
    print("=== 订单服务助手 ===")
    print("功能：查询订单 / 申请退款（输入'退出'结束）\n")

    
    # 主循环
    while True:
        # 首次获取用户输入
        user_input = input("您：")
        if user_input.lower() in ["退出", "q", "quit"]:
            print("系统：感谢使用，再见！")
            return
        
        current_state["user_input"] = user_input
        current_state["history"].append(f"您：{user_input}")
        
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        # 处理初始请求
        current_state = app.invoke(current_state, config=config)
        
        # 显示系统回复
        if current_state["history"]:
            last_msg = current_state["history"][-1]
            if last_msg.startswith("系统："):
                print(last_msg)
        # 如果需要追问，获取用户回应
        if current_state["need_followup"]:
            user_input = input("您：")
            if user_input.lower() in ["退出", "q", "quit"]:
                print("系统：感谢使用，再见！")
                break
            
            # 更新状态
            current_state["user_input"] = user_input
            current_state["history"].append(f"您：{user_input}")
            
            # 处理追问回应
            current_state = app.invoke(Command(resume=current_state), config=config)
            
            # 显示系统回复
            if current_state["history"]:
                last_msg = current_state["history"][-1]
                if last_msg.startswith("系统："):
                    print(last_msg)
        
        # 如果流程结束，重置状态并等待下一次查询
        if current_state["current_node"] == "call_tool" or not current_state["need_followup"]:
            # 重置状态
            current_state["followup_count"] = 0
            current_state["order_id"] = None
            current_state["refund_reason"] = None
            current_state["need_followup"] = False
            current_state["followup_question"] = None
            
          

if __name__ == "__main__":
    run_conversation()