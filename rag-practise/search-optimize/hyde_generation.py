from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
import os
import dotenv
dotenv.load_dotenv()


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 

# 加载文档并构建向量数据库
loader = TextLoader(f"{data_dir}/simple.txt", encoding='utf-8')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh") 
vectordb = Milvus.from_documents(documents=splits, embedding= embed_model)
# HyDE文档生成模板
template = """请撰写一段与以下问题相关的山西旅游景点的问题：
问题：{question}
内容："""
prompt_hyde = ChatPromptTemplate.from_template(template)
# 初始化模型
llm = ChatDeepSeek(model="deepseek-chat") 
# 创建生成假设文档的链
generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser()
)
# 示例问题
question = "山西主要旅游景点有哪些？"
# 生成假设文档
generated_doc = generate_docs_for_retrieval.invoke({"question": question})
print("hyde doc:", generated_doc)
# 初始化向量存储检索器
retriever = vectordb.as_retriever()
# 检索相关文档
retrieval_chain = generate_docs_for_retrieval | retriever
retrieved_docs = retrieval_chain.invoke({"question": question})
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n文档 {i}:")
    print(doc.page_content)
# 最终回答生成模板
answer_template = """根据以下内容回答问题：
{context}
问题：{question}
回答："""
answer_prompt = ChatPromptTemplate.from_template(answer_template)
# 创建最终的问答链
final_rag_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)
# 获取最终答案
final_answer = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
print(final_answer)