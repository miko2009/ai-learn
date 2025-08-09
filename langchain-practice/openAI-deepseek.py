# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam
import requests

load_dotenv(override=True)
from langchain.chat_models import init_chat_model
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    # 使用强类型消息参数
    messages = [
        ChatCompletionSystemMessageParam(role="system", content="Hello, world!"),
        ChatCompletionUserMessageParam(role="user", content="Hello, world!"),
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    response = client.chat.completions.create(model="deepseek-chat", messages = messages)
    print(response.choices)
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
