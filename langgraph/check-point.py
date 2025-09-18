from typing import Annotated, Sequence
from pydantic import BaseModel, Field
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # ←← 

# === 1. 定义状态 Schema (Pydantic) ===
class AgentState(BaseModel):
    messages: Annotated[Sequence[str], operator.add] = Field(default_factory=list)
    step_count: int = 0


# === 2. 节点函数 ===
def node1(state: AgentState) -> dict:
    print(f"[Node1] 当前 step_count: {state.step_count}")
    new_message = f"Hello from node1 at step {state.step_count + 1}"
    return {
        "messages": [new_message],
        "step_count": state.step_count + 1,
    }


def node2(state: AgentState) -> dict:
    print(f"[Node2] 当前 step_count: {state.step_count}")
    new_message = f"Goodbye from node2 at step {state.step_count + 1}"
    return {
        "messages": [new_message],
        "step_count": state.step_count + 1,
    }


# === 3. 构建图 ===
builder = StateGraph(AgentState)

builder.add_node("node1", node1)
builder.add_node("node2", node2)

builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

# 编译时必须传入 checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)  # ←← 启用 checkpointer

# === 4. 执行图并传入 config（包含 thread_id）===
config = {"configurable": {"thread_id": "123"}}
result = graph.invoke(
    {"messages": ["Initial input"], "step_count": 0},
    config=config
)

print("\n 最终输出:")
print(result)

# === 5. 获取状态历史（快照）===
print("\n 状态变更历史（快照）:")
state_history = list(graph.get_state_history(config))
for state in state_history:
    print(state)
