import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pymilvus import MilvusClient
import pymilvus.model as pymilvus_model
from glob import glob
from tqdm import tqdm
import json
import requests

load_dotenv(verbose=True)
collection_name = "mfd_rag_collection"
def send_messages(messages, tools):
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice = 'auto'
    )
    return response
def get_weather(location):
    api_key = os.getenv("WEATHER_API_KEY")
    api_host = "https://api.openweathermap.org/data/2.5/weather"
    print(f"Getting weather for {location}")
    param = {
        "appid": api_key,
        "q": location,
        "units": "metric",
    }
    response = requests.get(api_host, params=param)
    print(response.json())
    return json.dumps(response.json())

def main():
    available_tools = {
        "get_weather": get_weather,
    }


    SYSTEM_PROMPT = """
        you are a weather assistant，base on the user message, plz use a sentence to summary the city weather。
    """
    USER_PROMPT = "how is the xiamen's weather today? ？  you will receive a weather report {weather_json} from tool, plz format it to a sentence description."
    messages = [
        ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=USER_PROMPT),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of an location, the user should supply a location first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city",
                        }
                    },
                    "required": ["location"]
                },
            }
        },
    ]
    response = send_messages(messages, tools)
    print(response.choices[0].message)
    response_message = response.choices[0].message
    # **ReAct模式：处理工具调用**
    if response_message.tool_calls:  # 如果模型决定调用工具
        print("Agent: 决定调用工具...")
        messages.append(response_message)  # 将工具调用信息添加到对话历史

        tool_outputs = []
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            # 确保参数是合法的JSON字符串，即使工具不要求参数，也需要传递空字典
            function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

            print(f"Agent Action: 调用工具 '{function_name}'，参数：{function_args}")

            # 查找并执行对应的模拟工具函数
            if function_name in available_tools:
                tool_function = available_tools[function_name]
                tool_result = tool_function(**function_args)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": str(tool_result)  # 工具结果作为字符串返回
                })
            # else:
            #     error_message = f"错误：未知的工具 '{function_name}'"
            #     print(error_message)
            #     tool_outputs.append({
            #         "tool_call_id": tool_call.id,
            #         "role": "tool",
            #         "content": error_message
            #     })
        print(tool_outputs)
        messages.extend(tool_outputs) # 将工具执行结果作为 Observation 添加到对话历史

        result = send_messages(messages, tools)
        print(result.choices[0].message.content)

if __name__ == '__main__':
    main()