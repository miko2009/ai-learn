# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from pydantic import BaseModel, Field
from torch.fx.proxy import TracerBase
load_dotenv(override=True)
from langchain.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda  # 导入RunnableLambda


class Engineer(BaseModel):
    skills: list[str] = Field(description="技术栈列表，如 'java, c++ '")
    work_experience: int = Field(description="工作时长，如 10 ")

class FirstOutput(BaseModel):
    jd: str = Field(description="工作要求")
    skills: list[str] = Field(description="技术栈列表，如 'java, c++ '")
    work_experience: int = Field(description="工作时长，如 10 ")


def debug_mode(x):
    """调试函数，打印中间结果"""
    print("\n~~~~调试信息~~~~")
    print(f"类型: {type(x)}")
    print(f"内容: {x}")
    return x


def transform_data(inputs):
    """修改转换函数，接收包含多个参数的字典"""
    return {
        "jd": inputs["first_output"].jd,
        "skills": inputs["first_output"].skills,
        "work_experience": inputs["first_output"].work_experience,
        "output_format_instruction": inputs["output_format_instruction"]  # 从输入字典中获取
    }

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    schemas = [
        ResponseSchema(name = "skill_evaluation", description = "技能是否符合"),
        ResponseSchema(name = "work_exp_evaluation", description = "工作年限是否符合"),
    ]
    model = ChatDeepSeek(model = "deepseek-chat")
    first_parser = PydanticOutputParser(pydantic_object=FirstOutput)
    format_instruction = first_parser.get_format_instructions()
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "提取用户问题的核心信息,按json: 按 {format_instruction}输出"),
        ("user", "{user_query} ")])

    retriever_chain = retriever_prompt | model | first_parser

    result = retriever_chain.invoke({
        "user_query": "候选人信息：名字张三,年龄:28,技能 c++ jave 工作了 9 年。工作要求：8年以前的的php 工作经验",
        "format_instruction": format_instruction
    })
    parser = StructuredOutputParser.from_response_schemas(schemas)
    output_format_instruction = parser.get_format_instructions()
    interview_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个程序员招聘专家, 请根据工作要求和候选人情况,输出评估报告, 格式为json: {output_format_instruction}"),
        ("user", "工作要求: {jd},候选人工作skill: {skills}, 候选人work_experience: {work_experience}")
    ])  
    interview_chain = interview_prompt | debug_mode | model | parser
       # 组合链：使用RunnableLambda包装转换函数，传递多个参数
    overall_chain = (
        {
            "first_output": retriever_chain,  # 第一个链的输出
            "output_format_instruction": lambda _: output_format_instruction  # 传递格式指令
        }
        | RunnableLambda(transform_data)  # 转换数据
        | interview_chain  # 评估链
    )
    result = overall_chain.invoke({
        "user_query":  "候选人信息：名字张三,年龄:28,技能 c++ java 工作了 9 年。工作要求：8年以前的的php 工作经验",
        "format_instruction": format_instruction,
    })
    print(result)

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
