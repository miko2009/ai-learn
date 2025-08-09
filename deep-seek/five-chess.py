import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

load_dotenv(verbose=True)
def main():
    prompt = 'web开发生成一个五子棋游戏的代码，并保存在一个html 文件中'
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
    # 使用强类型消息参数
    # messages = [
    #     {"role": "system", "content":"你是一个经验丰富web开发程序员, 擅长用 HTML/CSS/JavaScript 编写游戏."},
    #     {"role": "user", "content": prompt}
    # ]
    # 使用强类型消息参数
    messages = [
        ChatCompletionSystemMessageParam(role="system", content="你是一个经验丰富web开发程序员, 擅长用 HTML/CSS/JavaScript 编写游戏."),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        stream=False
    )
    print(response)
    # 提取生成的 HTML 内容
    if response.choices and len(response.choices) > 0:
        html_content = response.choices[0].message.content

        # 保存到文件
        with open("gomoku.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("五子棋游戏已保存为 gomoku.html")
    else:
        print("未收到有效响应")
if __name__ == '__main__':
    main()