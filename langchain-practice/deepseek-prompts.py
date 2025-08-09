# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
from dotenv import load_dotenv
from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam
import requests

load_dotenv(override=True)
from langchain.chat_models import init_chat_model
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.prompts import ChatPromptTemplate
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    model = init_chat_model(model = "deepseek-chat", model_provider = "deepseek")
    question = "1 + 1 是否 大于 2"
    prompt_template = ChatPromptTemplate([
        ("system", "you are an assistant, plz answer the uer question "),
        ("user", "this is the question: {topic}, plz use yes or no to answer the question "),
    ])
    chain = prompt_template | model | BooleanOutputParser()
    result = chain.invoke(question)
    print(result)
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
