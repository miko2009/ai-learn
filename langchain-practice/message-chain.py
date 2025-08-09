# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
load_dotenv(override=True)
from langchain.chat_models import init_chat_model

def debug_mode(x):
    print(x)
    return x

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    model = init_chat_model(model = "deepseek-chat", model_provider = "deepseek")
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content= "你叫小智， 是一名ai小助手"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    parser = StrOutputParser()
    basic_chain = chat_prompt | model | parser
    msg_list = [
        HumanMessage(content="我叫小明, nice to meet you!"),
        AIMessage(content="我叫小智,nice to meet you!"),
        HumanMessage(content="我叫什么名字")
    ]

    msg_list = []
    while True:
        user_query = input("你：")
        if user_query == "exit":
            break
        msg_list.append(HumanMessage(content=user_query))
        ai_reply = basic_chain.invoke({"messages": msg_list})
        print("ai:",ai_reply)
        msg_list.append(AIMessage(content=ai_reply))
        msg_list = msg_list[-50:]
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
