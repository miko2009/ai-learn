import json
from langchain.chains.router.multi_prompt_prompt import (
    MULTI_PROMPT_ROUTER_TEMPLATE
)
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain.chains.llm import LLMChain
from langchain.chains.router.llm_router import (
    LLMRouterChain,
    RouterOutputParser
)
from langchain.chains import ConversationChain
from langchain.chains.router import MultiPromptChain
import dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda

dotenv.load_dotenv()

llm = Tongyi(
    temperature=0.1,
)
# 售前咨询模板
presales_prompt_tpl = PromptTemplate.from_template(
    '你是一位专业的售前顾问，擅长产品介绍、方案推荐和商务咨询。'
    '你需要热情、专业地回答客户的产品咨询、价格询问、功能介绍等售前问题。'
    '请使用中文帮我解答下列售前咨询问题：\n{input}'
)

# 售后服务模板
aftersales_prompt_tpl = PromptTemplate.from_template(
    '你是一位耐心的售后服务专员，擅长解决客户的使用问题、技术支持和投诉处理。'
    '你需要耐心、细致地帮助客户解决产品使用中遇到的问题，提供技术支持和服务指导。'
    '请使用中文帮我解答下列售后服务问题：\n{input}'
)
default_prompt = PromptTemplate.from_template(
    '你是一位耐心的服务专员，擅长解决客户的使用问题、技术支持和投诉处理。'
    '请使用中文帮我解答下列服务问题：\n{input}'
)
prefix_chain = presales_prompt_tpl | llm | StrOutputParser()
end_chain = aftersales_prompt_tpl | llm | StrOutputParser()
default_chain = default_prompt | llm | StrOutputParser()


# 创建路由函数
def route_question(input_dict):
    main_prompt = PromptTemplate.from_template("""
        你是一个问题分类小助手, 可以根据客户的问题{input}准确的分成售前或者售后问题。返回以下json 格式:
        {{ "is_presales": 售前问题 true, "is_aftersales": 售后问题 }}
    """)
    router_chain =  main_prompt | llm | JsonOutputParser()
    question = input_dict["input"]
    intent = router_chain.invoke({"input": question})
    print(f"识别意图: {intent}")
    intent_dict = json.loads(json.dumps(intent))
    print(intent_dict)
    
    if intent_dict["is_presales"]:
        return prefix_chain.invoke({"input": question})
    elif intent_dict["is_aftersales"]:
        return end_chain.invoke({"input": question})
    else:
        return default_chain.invoke({"input": question})


# 测试代码
if __name__ == "__main__":
    # print("=== 方法一：自定义路由函数 ===")
    router_chain = RunnablePassthrough() | RunnableLambda(route_question)
    # # 测试售前问题
    # print("\n--- 售前咨询测试 ---")
    result1 = router_chain.invoke({"input": "你们的产品有什么功能？价格是多少？"})
    print(f"回答: {result1}")

    print("\n=== 售后服务测试 ===") 
    print(router_chain.invoke({"input": "我的产品出现故障了，无法正常启动，该怎么办？"}))


