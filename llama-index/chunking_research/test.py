from cgitb import text
from pydoc import doc
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# 使用SimpleDirectoryReader加载目录中的文档
documents = SimpleDirectoryReader(input_files=[f"{current_dir}/data/memory.txt"]).load_data()
# 检查加载的文档类型
print(documents)

for i, doc in enumerate(documents):
    print(doc.id_)

# 创建SentenceSplitter实例，设置块大小和重叠
custom_splitter = SentenceSplitter(
    separator="。",  # 使用句号作为分割符（适用于中文）
    chunk_size=200,
).split_text

# 使用SentenceWindowNodeParser，传入SentenceSplitter
window_parser = SentenceWindowNodeParser.from_defaults(
    window_size=2,
    sentence_splitter=custom_splitter
)
# nodes_1 = custom_splitter.get_nodes_from_documents([Document(
#     text = "efg。"
# )])
# print(nodes_1)
nodes = window_parser.get_nodes_from_documents([Document(
    text = 'abc'
)])
# 获取分割后的节点

# 打印节点信息
for i, node in enumerate(nodes):
    print(f"node {i}")
    print(f"window: {node.metadata.get('window', 'n/a')}")
    print(f"original text: {node.text}")
    print("----")