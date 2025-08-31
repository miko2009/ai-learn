import logging
from langchain.retrievers import RePhraseQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
import os
import dotenv
dotenv.load_dotenv()

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 

# 设置日志记录
logging.basicConfig()
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)
# 加载游戏文档数据
loader = TextLoader(f"{data_dir}/simple.txt", encoding='utf-8')
data = loader.load()
# 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
# 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Milvus.from_documents(documents=all_splits, embedding= embed_model)
# 设置RePhraseQueryRetriever
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
retriever_from_llm = RePhraseQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm # 使用DeepSeek模型做重写器
)
# 示例输入：游戏相关查询
query = "我来山西旅游了，但是我不知道去哪里，花费怎么样?"
# 调用RePhraseQueryRetriever进行查询重写
docs = retriever_from_llm.invoke(query)
print(docs)