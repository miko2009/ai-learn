from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_community.llms import Tongyi
import dotenv
dotenv.load_dotenv()
# 初始化
parser = CommaSeparatedListOutputParser()
llm = Tongyi(temperature=0)

# 直接构建提示并调用
def simple_list_generation(category):
    print("parser:", parser.get_format_instructions())
    # 手动构建提示
    prompt = f"""请列出5个{category}的例子。
        {parser.get_format_instructions()}"""
    
    # 直接调用LLM
    response = llm.invoke(prompt)
    
    # 解析结果
    return parser.parse(response)

# 使用
fruits = simple_list_generation("水果")
print("水果列表:", fruits)

languages = simple_list_generation("编程语言")
print("编程语言列表:", languages)
