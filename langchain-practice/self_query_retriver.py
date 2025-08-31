import logging
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.self_query.base import SelfQueryRetriever, AttributeInfo
from langchain_milvus import Milvus
from langchain_core.documents import Document
import dotenv
import os

dotenv.load_dotenv()
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 
# 设置日志记录
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
documents = [
    Document(
        page_content="Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言。",
        metadata={"language": "Python", "difficulty": "beginner", "topic": "programming", "year": 2023}
    ),
    Document(
        page_content="Java 是一种跨平台的面向对象编程语言，广泛用于企业级应用开发。",
        metadata={"language": "Java", "difficulty": "intermediate", "topic": "programming", "year": 2022}
    ),
    Document(
        page_content="机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。",
        metadata={"language": "N/A", "difficulty": "advanced", "topic": "ai", "year": 2023}
    ),
    Document(
        page_content="React 是一个用于构建用户界面的 JavaScript 库，由 Facebook 开发。",
        metadata={"language": "JavaScript", "difficulty": "intermediate", "topic": "web", "year": 2021}
    ),
    Document(
        page_content="Milvus 是一个开源向量数据库，专为 AI 应用和向量相似度搜索设计。",
        metadata={"language": "N/A", "difficulty": "advanced", "topic": "database", "year": 2023}
    )
]
# 加载游戏相关文档并构建向量数据库
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=embed_model,
    connection_args={"host": "localhost", "port": "19530"},  # Milvus连接参数
    collection_name="tech_docs",  # 指定集合名称
    drop_old=True  # 可选：如果集合已存在则删除旧数据
)

# 2. 定义文档元数据的结构信息
metadata_field_info = [
    AttributeInfo(
        name="language",
        description="文档讨论的编程语言，如 Python、Java，如果不涉及特定语言则为 'N/A'",
        type="string",
    ),
    AttributeInfo(
        name="difficulty",
        description="文档内容的难度级别",
        type="string",  # 可以是 'beginner', 'intermediate', 'advanced'
    ),
    AttributeInfo(
        name="topic",
        description="文档的主题类别",
        type="string",  # 可以是 'programming', 'ai', 'web', 'database' 等
    ),
    AttributeInfo(
        name="year",
        description="文档内容相关的年份",
        type="integer",
    ),
]
document_content_description = "关于编程语言、人工智能和数据库的技术文档"

# 通过MultiQueryRetriever 生成多角度查询
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
retriever_from_llm = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True  # 输出解析过程，便于调试
)
results = retriever_from_llm.get_relevant_documents("2023年发布的适合初学者的编程文档")
for i, doc in enumerate(results, 1):
    print(f"\n结果 {i}:")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")