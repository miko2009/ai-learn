import logging
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever # 多角度查询检索器
from langchain.prompts import PromptTemplate
from langchain_milvus import Milvus
import dotenv
import os
from langchain_core.output_parsers import BaseOutputParser
from typing import List

dotenv.load_dotenv()
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 
# 设置日志记录
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
# 加载游戏相关文档并构建向量数据库
loader = TextLoader(f"{data_dir}/simple.txt", encoding='utf-8')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Milvus.from_documents(documents=splits, embedding= embed_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一个资深的山西本地导游, 可以帮游客推荐旅游景点, 及景点收费和交通。
                用户原始问题：{question}
                请给出3个不同的查询，每个占一行。""",
)
class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # 过滤空行
output_parser = LineListOutputParser()

# 通过MultiQueryRetriever 生成多角度查询
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_chain = QUERY_PROMPT | llm | output_parser
# 使用自定义提示模板的MultiQueryRetriever
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(), 
    llm_chain=llm_chain, 
    parser_key="lines"
)
query = "山西有什么好去处？"
# 调用RePhraseQueryRetriever进行查询分解
docs = retriever.invoke(query)
print(docs)