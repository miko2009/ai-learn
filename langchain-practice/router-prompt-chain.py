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

prompt_infos = [
    {
        'name': 'presales',
        'description': '用于处理售前咨询，包括产品介绍、价格询问、功能说明、方案推荐等',
        'prompt_template': presales_prompt_tpl,
    },
    {
        'name': 'aftersales',
        'description': '用于处理售后服务，包括使用问题、技术支持、故障排除、投诉处理等',
        'prompt_template': aftersales_prompt_tpl,
    },
]

# 生成键为模板名称、值为Chain的字典
destination_chains = {}
for p_info in prompt_infos:
    name = p_info['name']
    prompt = p_info['prompt_template']
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
# 将模板名称和模板描述通过MULTI_PROMPT_ROUTER_TEMPLATE生成模板
destinations = [f'{p["name"]}: {p["description"]}'
                for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
print(router_template)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=['input'],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# 这里创建了一个default_chain
# 为了防止提的问题类型并没有包含在prompt_infos中
default_chain = ConversationChain(llm=llm, output_key='text')
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

# 测试售前咨询问题
print("=== 售前咨询测试 ===")
print(chain.invoke("你们的产品有什么功能？价格是多少？"))

# print("\n=== 售后服务测试 ===")
# print(chain.run("我的产品出现故障了，无法正常启动，该怎么办？"))

# print("\n=== 其他问题测试 ===")
# print(chain.run("今天天气怎么样？"))