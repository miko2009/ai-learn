# This is a sample Python script.
import json
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os

import requests
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, JsonOutputKeyToolsParser
from langchain_core.messages.tool import tool_call
from langchain_core.prompts import PromptTemplate
import pandas as pd
from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers import StrOutputParser
load_dotenv(override=True)
from langchain.chat_models import init_chat_model

@tool(description="Gets the current weather for a specified location")
def get_weather(loc):
    api_key = os.getenv("WEATHER_API_KEY")
    api_host = "https://api.openweathermap.org/data/2.5/weather"
    print(f"Getting weather for {loc}")
    param = {
        "appid": api_key,
        "q": loc,
        "units": "metric",
    }
    response = requests.get(api_host, params=param)
    return json.dumps(response.json())

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    model = init_chat_model(model = "deepseek-chat", model_provider = "deepseek")

    df = pd.read_csv("~/Downloads/global_cities_data.csv")
    table_tool = PythonAstREPLTool(locals=({"df": df}))
    tools = [get_weather]
    llm_with_tools = model.bind_tools(tools)
    parser = JsonOutputKeyToolsParser( key_name=get_weather.name, first_tool_only=True)
    # response = llm_with_tools.invoke("我有一张表, 名为 'df', 请帮我计算 GDP_Billion_USD 字段的平均值")
    tool_chain = llm_with_tools | parser | get_weather

    response_template = PromptTemplate.from_template(
        "you will receive a weather report {weather_json}, plz format it to a sentence description.",
    )
    out_chain = response_template | model | StrOutputParser()
    full_chain = tool_chain | out_chain
    result = full_chain.invoke("what is the xiamen weather")
    print(result)

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
