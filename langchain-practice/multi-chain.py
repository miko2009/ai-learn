# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
load_dotenv(override=True)
from langchain.chat_models import init_chat_model

def debug_mode(x):
    print(x)
    return x

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    model = init_chat_model(model = "deepseek-chat", model_provider = "deepseek")

    news_prompt = PromptTemplate.from_template(
        "base on the following context, generate a brief news and the news(less than 100): \n\ntitle:{title}"
    )
    news_chain = news_prompt | model
    schemas = [
        ResponseSchema(name = "time", description = "occurrence time"),
        ResponseSchema(name = "location", description = "occurrence location"),
        ResponseSchema(name = "event", description = "mode detail"),
    ]
    parser = StructuredOutputParser.from_response_schemas(schemas)

    summary_prompt = PromptTemplate.from_template(
        "summary of the news and return the follow JSON: \n\n{news}\n\n{format_instructions}"
    )
    summary_chain = summary_prompt.partial(format_instructions = parser.get_format_instructions()) | model | parser
    debug_chain = RunnableLambda(debug_mode)
    full_chain = news_chain | debug_chain | summary_chain
    result = full_chain.invoke({"title": "苹果公司在加州发布最新的ai 芯片"})
    print(result)
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
