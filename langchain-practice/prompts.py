from langchain_core.prompts import PromptTemplate


prompt = PromptTemplate.from_template(
    "Hello, I am a {model_name}. How can I help you today?"
)
print(prompt.format(model_name="chatbot"))

complex_prompt = PromptTemplate(
    input_variables=["topic", "audience", "tone"],
    template="""
    请为{audience}写一篇关于{topic}的文章。
    写作风格应该是{tone}的。
    
    文章要求：
    - 内容准确且有用
    - 结构清晰
    - 适合目标受众
    """
)

formatted_prompt = complex_prompt.format(
    topic="人工智能",
    audience="初学者",
    tone="通俗易懂"
)
print(formatted_prompt)


# f-string 风格
f_string_prompt = PromptTemplate.from_template(
    "分析以下{data_type}数据：\n{data}\n\n请提供{analysis_type}分析。"
)

result = f_string_prompt.format(
    data_type="销售",
    data="Q1销售额: 100万, Q2销售额: 120万",
    analysis_type="趋势"
)
print(result)


from langchain_core.prompts import ChatPromptTemplate

# 创建包含 system 和 human 消息的聊天模板
chat_prompt = ChatPromptTemplate([
    ("system", "You are a helpful AI assistant named {assistant_name}."),
    ("human", "Hello, my name is {user_name}. {question}")
])

# 格式化消息
messages = chat_prompt.format_messages(
    assistant_name="Claude",
    user_name="Alice", 
    question="What's the weather like today?"
)

for message in messages:
    print(f"{message.type}: {message.content}")



prompt = PromptTemplate(
    input_variables=["name", "age"],
    template="你好，我是{name}，今年{age}岁"
)

# 查看输入 schema
print("输入 Schema:")
print(prompt.input_schema.schema())

# 查看输出 schema
print("输出 Schema:")
print(prompt.output_schema.schema())