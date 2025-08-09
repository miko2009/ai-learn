# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv(override=True)
from langchain.chat_models import init_chat_model
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.prompts import ChatPromptTemplate
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    schemas = [
        ResponseSchema(name = "name", description = "username"),
        ResponseSchema(name = "age", description = "age"),
    ]
    parser = StructuredOutputParser.from_response_schemas(schemas)
    prompt = PromptTemplate.from_template(
        "base on the following context, return the JSON: \n{input}\n\n{format_instructions}"
    )

    model = init_chat_model(model = "deepseek-chat", model_provider = "deepseek")
    question = {"input": "user name is brucelee, he is an engineer, 25 years old ", }

    chain = prompt.partial(format_instructions = parser.get_format_instructions()) | model | parser
    result = chain.invoke(question)
    print(result)
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
