from math import sin
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)

from llama_index.embeddings.openai import OpenAIEmbedding 

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 

def load_data_with_llama_index(single_file):
    documents = SimpleDirectoryReader(input_files=[single_file]).load_data()
    splitter = SemanticSplitterNodeParser(
        buffer_size=3,  # 缓冲区大小
        breakpoint_percentile_threshold=90, # 断点百分位阈值
        embed_model=OpenAIEmbedding()     # 使用的嵌入模型
    )
    # 创建基础句子分块器（作为对照）
    base_splitter = SentenceSplitter(
        # chunk_size=512
    )
    semantic_nodes = splitter.get_nodes_from_documents(documents)
    print(f"nodes_num: {len(semantic_nodes)}")
    for i, node in enumerate(semantic_nodes, 1):
        print(f"content:\n{node.text}")
    base_nodes = base_splitter.get_nodes_from_documents(documents)
    print(f"node len: {len(base_nodes)}")
    for i, node in enumerate(base_nodes, 1):
        print(f"\n--- NO. {i}---")
        print(f"content:\n{node.text}")
def chunk_with_langchain(single_file):
    loader = TextLoader(single_file)
    documents = loader.load()
    # 设置分块器，指定块的大小为50个字符，无重叠
    text_splitter = CharacterTextSplitter(
        chunk_size=500,  # 每个文本块的大小为100个字符
        chunk_overlap=0,  # 文本块之间没有重叠部分
    )
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- 第 {i} 个文档块 ---")
        print(f"内容: {chunk.page_content}")
        print(f"元数据: {chunk.metadata}")
def chunk_with_langchain_recursiveCharacter(single_file):
    loader = TextLoader(single_file)
    documents = loader.load()
        # 定义分割符列表，按优先级依次使用
    separators = ["\n\n", "。", "，", " "] # . 是句号，， 是逗号， 是空格
    # 创建递归分块器，并传入分割符列表
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=10,
        separators=separators
    )
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n---num: {i} ---")
        print(f"content: {chunk.page_content}")
        print(f"metadata: {chunk.metadata}")
if __name__ == "__main__":
    single_file = f"{data_dir}/simple.txt"
    # chunk_with_langchain(single_file)
    # print("____chunk end___________\n")
    chunk_with_langchain_recursiveCharacter(single_file)
    print("________recursive chunk end___________\n")
    # load_data_with_llama_index(data_dir)
    # print("________llama index end___________\n")