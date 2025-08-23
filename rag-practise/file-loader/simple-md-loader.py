from pydoc import doc
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from unstructured.partition.md import partition_md
from llama_index.core import SimpleDirectoryReader
import os


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 

def load_data_with_llama_index(data_dir):
    simple_file_loader = SimpleDirectoryReader(input_dir=data_dir, required_exts=['.md'])
    docs = simple_file_loader.load_data()
    print("llama index docs: ", docs)
    # 通过vars函数查看所有可用的元数据
    for i, element in enumerate(docs):
        print(f"\n--- Element {i+1} ---")
        print(f"类型: {type(element)}")

    return docs


def load_data_with_langchain_loader(data_dir):
    loader = UnstructuredMarkdownLoader(f"{data_dir}/wukong-md.md")
    docs = loader.load()
    print("langchain docs: ", docs)
    # 通过vars函数查看所有可用的元数据
    for i, element in enumerate(docs):
        print(f"\n--- Element {i+1} ---")
        print(f"类型: {type(element)}")

    return docs


def load_data_with_unstructured(data_dir):
    docs = partition_md(f"{data_dir}/wukong-md.md", strategy="hi_res")
    print("Unstructured docs: ", docs)

    # 通过vars函数查看所有可用的元数据
    for i, element in enumerate(docs):
        print(f"\n--- Element {i+1} ---")
        print(f"类型: {type(element)}")
        print(f"元素类型: {element.__class__.__name__}")
        print(f"文本内容: {element.text}")
        if hasattr(element, 'metadata'):
            print("元数据:")
            metadata_dict = element.metadata.__dict__
            for key, value in metadata_dict.items():
                if not key.startswith('_') and value is not None:  
                    print(f"  {key}: {value}")
    return docs


if __name__ == "__main__":
    load_data_with_unstructured(data_dir)
    print("________unstructured end___________\n")
    load_data_with_langchain_loader(data_dir)
    print("________langchin end___________\n")
    load_data_with_llama_index(data_dir)
    print("________llama index end___________\n")