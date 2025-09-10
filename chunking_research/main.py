import os
from pydoc import doc
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser, TextSplitter, TokenTextSplitter
import dotenv
dotenv.load_dotenv()

Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

def load_file(data_dir):
    simple_file_loader = SimpleDirectoryReader(input_files=[f"{data_dir}/memory.txt"])
    docs = simple_file_loader.load_data()
    return docs

def chunk_by_sentence(docs):
    sentence_splitter = SentenceSplitter(
        chunk_size= 1000,
        separator="。",  # 中文主分割符（句号）
        chunk_overlap=30,  # 相邻片段重叠30个字符
        # 补充其他中文句末符号（避免漏分割）
    )
    nodes = sentence_splitter.get_nodes_from_documents(docs)

    index = VectorStoreIndex.from_documents(
        nodes, 
    )
    # 创建查询引擎（设置返回 top3 最相关的分割片段）
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        verbose=True  # 打印检索过程日志
    )
    response = query_engine.query("工作记忆的结构与容量限制")
    for i, source_node in enumerate(response.source_nodes, 1):
        print(f"  {i}. 片段内容：{source_node.node.text}")
        print(f"     相关性得分：{source_node.score:.4f}") 

    
def chunk_by_sentence_window(docs):

    sentence_splitter = SentenceSplitter(
        separator="。",  # 中文主分割符（句号）
        chunk_overlap=30,  # 相邻片段重叠30个字符
        # 补充其他中文句末符号（避免漏分割）
    ).split_text
    window_parse = SentenceWindowNodeParser.from_defaults(
        window_size=3,  # 前后各3个句子
        window_metadata_key="window_context",  # 上下文存储的元数据键
        original_text_metadata_key="original_sentence",  # 原始句子存储的元数据键
        sentence_splitter=sentence_splitter,
    )

    nodes = window_parse.get_nodes_from_documents(docs)

    index = VectorStoreIndex.from_documents(
        nodes, 
    )
    # 创建查询引擎（设置返回 top3 最相关的分割片段）
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        verbose=True  # 打印检索过程日志
    )
    response = query_engine.query("工作记忆的结构与容量限制")
    print(f"【AI回答】：{response}")
    print("【检索到的相关片段】：")
    for i, source_node in enumerate(response.source_nodes, 1):
        print(f"  {i}. 片段内容：{source_node.node.text}")
        print(f"     相关性得分：{source_node.score:.4f}") 

def chunk_by_token(docs):
    token_splitter = TokenTextSplitter(
        chunk_size=1000,  # 单个片段最多字符
        chunk_overlap=30,  # 相邻片段重叠30个字符
    )
    nodes = token_splitter.get_nodes_from_documents(docs)

    index = VectorStoreIndex.from_documents(
        nodes, 
    )
    # 创建查询引擎（设置返回 top3 最相关的分割片段）
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        verbose=True  # 打印检索过程日志
    )
    response = query_engine.query("工作记忆的结构与容量限制")
    print(f"【AI回答】：{response}")
    print("【检索到的相关片段】：")
    for i, source_node in enumerate(response.source_nodes, 1):
        print(f"  {i}. 片段内容：{source_node.node.text}")
        print(f"     相关性得分：{source_node.score:.4f}") 

def main():
    data_dir = f"{current_dir}/data"
    docs = load_file(data_dir)
    chunk_by_sentence(docs)
    print("-------by sentence end -------")
    chunk_by_sentence_window(docs)
    print("-------by sentence window end -------")
    chunk_by_token(docs)
    print("-------by token end -------")
    pass

if __name__ == "__main__":
    main()