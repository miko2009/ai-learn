import os
import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents import AgentExecutor, create_tool_calling_agent

# 加载环境变量（如通义千问 API Key）
load_dotenv(override=True)

# ---------------------- 1. 定义工具（获取系统时间） ----------------------
@tool(description="获取当前系统的本地时间，格式为'年-月-日 时:分:秒'。当用户问题中包含时间性词汇（如昨天、今天、明天），需要推断具体日期时，必须调用此工具获取当前时间作为基准。")
def get_sys_time() -> str:
    """获取当前本地时间的字符串格式"""
    current_time = datetime.datetime.now()
    return current_time.strftime("%Y-%m-%d %H:%M:%S")

# 工具列表（后续可扩展其他工具）
tools = [get_sys_time]

# ---------------------- 2. 初始化 LLM（支持工具调用） ----------------------
llm = ChatTongyi(
    model_name="qwen-turbo",  # 确保模型支持工具调用（通义千问 turbo 及以上版本支持）
    temperature=0.7,
    streaming=True
)

# 将工具格式化为 LLM 可识别的 OpenAI 函数格式（通义千问兼容此格式）
llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(tool) for tool in tools]
)

# ---------------------- 3. 定义提示词（明确工具调用逻辑） ----------------------
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""你是专业的运营小助手，必须严格按以下步骤处理含时间词汇的问题：
    步骤1：调用get_sys_time获取当前时间（格式'YYYY-MM-DD HH:MM:SS'），**必须先执行这一步**。
    步骤2：从当前时间中提取日期部分（'YYYY-MM-DD'），作为current_time参数；确定用户的相对时间词（如'昨天'）作为relative_term参数，调用calculate_target_date工具。
    步骤3：根据calculate_target_date返回的具体日期，生成回答（必须包含该日期）。
    
    示例流程：
    用户问"昨天的订单" → 调用get_sys_time得到"2024-10-05 10:00:00" → 调用calculate_target_date(current_time="2024-10-05", relative_term="昨天")得到"2024-10-04" → 回答"昨天（2024-10-04）的订单有..."
    
    注意：
    - 绝对不要自己计算日期，必须通过calculate_target_date工具
    - 非时间问题直接回答，不调用工具"""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


# ---------------------- 4. 创建工具调用 Agent（处理工具调用逻辑） ----------------------
# 创建工具调用 Agent：负责判断是否调用工具、执行工具、整合结果
agent = create_tool_calling_agent(
    llm=llm_with_tools,
    tools=tools,
    prompt=prompt
)

# 创建 Agent 执行器：管理 Agent 的运行流程（多轮工具调用+回答生成）
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 开启 verbose 可查看工具调用细节（便于调试）
    return_intermediate_steps=True # 是否返回中间步骤（工具调用记录等）
)

# ---------------------- 5. 多轮对话交互逻辑 ----------------------
def run_chatbot():
    print("运营小助手已启动（输入 'exit' 退出）\n")
    messages = []  # 存储对话历史（支持多轮上下文）
    
    while True:
        user_query = input("你：")
        if user_query.lower() == "exit":
            print("小助手：再见！")
            break
        
        # 将用户输入加入对话历史
        messages.append(HumanMessage(content=user_query))
        
        # 执行 Agent（自动处理工具调用和回答生成）
        response = agent_executor.invoke({"messages": messages})
        
        # 提取并打印回答
        ai_reply = response["output"]
        print(f"小助手：{ai_reply}")
        
        # 将 AI 回答加入对话历史（支持上下文关联）
        messages.append(AIMessage(content=ai_reply))
        
        # 限制对话历史长度（避免上下文过长，可选）
        if len(messages) > 10:  # 保留最近 10 条消息（5轮对话）
            messages = messages[-10:]

# ---------------------- 6. 启动聊天机器人 ----------------------
if __name__ == "__main__":
    run_chatbot()