import json
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



def is_prefix(intent):
    intent_dict = json.loads(json.dumps(intent))
    return intent_dict['intent']['is_presales']
def is_after(intent):
    intent_dict = json.loads(json.dumps(intent))
    return intent_dict['intent']['is_aftersales']

# 测试代码
if __name__ == "__main__":
    # print("=== 方法一：自定义路由函数 ===")
    main_chain = RunnableBranch(
        (is_prefix, prefix_chain),
        (is_after, end_chain),
        default_chain,
    )
    intent_prompt = PromptTemplate.from_template("""
        你是一个智能问题分类助手，需要根据客户的问题判断其类型：
        - 售前问题：包含产品介绍、价格询问、功能咨询、方案推荐、购买意向等内容；
        - 售后问题：包含产品故障、使用指导、技术支持、投诉处理、退换货等内容。

        请严格按照以下 JSON 格式输出分类结果（无需额外文字）：
        {{
            "is_presales": true/false,  // 是售前问题则为 true，否则为 false
            "is_aftersales": true/false // 是售后问题则为 true，否则为 false
        }}

        客户的问题：{input}
    """)
    intent_chain =  intent_prompt | llm | JsonOutputParser()
    all_chain = RunnablePassthrough.assign(intent=lambda x: intent_chain.invoke(x) ) | main_chain

    # print("\n=== 售qian服务测试 ===") 
    print(all_chain.invoke({"input": "A产品续航多久？"}))
    # print("\n=== 售后服务测试 ===") 
    print(all_chain.invoke({"input": "我的产品出现故障了，无法正常启动，该怎么办？"}))


