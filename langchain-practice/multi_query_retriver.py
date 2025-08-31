import logging
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever # 多角度查询检索器
from langchain_milvus import Milvus
import dotenv
import os

dotenv.load_dotenv()
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
# 设置日志记录
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
# 加载游戏相关文档并构建向量数据库
loader = TextLoader(f"{current_dir}/data/simple.txt", encoding='utf-8')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=embed_model,
    connection_args={"host": "localhost", "port": "19530"},  # Milvus连接参数
    collection_name="multi_retriver",  # 指定集合名称
    drop_old=True  # 可选：如果集合已存在则删除旧数据
)

# 通过MultiQueryRetriever 生成多角度查询
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), 
    llm=llm
)
query = "山西有什么好去处？"
# 调用RePhraseQueryRetriever进行查询分解
docs = retriever_from_llm.invoke(query)
print(docs)