from pydoc import doc
from langchain_community.document_loaders import PyPDFLoader
from unstructured.partition.pdf import partition_pdf
from llama_index.core import SimpleDirectoryReader
import os
import pymupdf


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 

def load_data_with_llama_index(data_dir):
    simple_file_loader = SimpleDirectoryReader(input_dir=data_dir, required_exts=['.pdf'])
    docs = simple_file_loader.load_data()
    print("llama index docs: ", docs)
    # 通过vars函数查看所有可用的元数据
    for i, element in enumerate(docs):
        print(f"\n--- Element {i+1} ---")
        print(f"类型: {element}")

    return docs


def load_data_with_langchain_loader(data_dir):
    loader = PyPDFLoader(f"{data_dir}/wukong-pdf.pdf")
    docs = loader.load()
    print("langchain docs: ", docs)
    # 通过vars函数查看所有可用的元数据
    for i, element in enumerate(docs):
        print(f"\n--- Element {i+1} ---")
        print(f"类型: {element}")

    return docs


def load_data_with_unstructured(data_dir):
    docs = partition_pdf(filename=f"{data_dir}/wukong-pdf.pdf")
    print("Unstructured docs: ", docs)

    # 通过vars函数查看所有可用的元数据
    for i, element in enumerate(docs):
        print(f"\n--- Element {i+1} ---")
        print(f"类型: {element}")
        print(f"文本内容: {element.text}")
        if hasattr(element, 'metadata'):
            print("元数据:")
            metadata_dict = element.metadata.__dict__
            for key, value in metadata_dict.items():
                if not key.startswith('_') and value is not None:  
                    print(f"  {key}: {value}")
    return docs

def load_data_with_pymupdf(data_dir): 
    doc = pymupdf.open(f"{data_dir}/wukong-pdf.pdf")
    print("=== PyMuPDF 基本信息提取 ===")
    print(f"文档页数: {len(doc)}")
    print(f"文档标题: {doc.metadata['title']}")
    print(f"文档作者: {doc.metadata['author']}")
    print(f"文档元数据: {doc.metadata}")  # 比Unstructured提供更多元数据
    # 遍历每一页
    for page_num, page in enumerate(doc):
        # 提取文本
        text = page.get_text()
        print(f"\n--- 第{page_num + 1}页 ---")
        print("文本内容:", text[:200])  # 显示前200个字符
        
        # 提取图片
        images = page.get_images()
        print(f"图片数量: {len(images)}")
        
        # 获取页面链接
        links = page.get_links()
        print(f"链接数量: {len(links)}")
        
        # 获取页面大小
        width, height = page.rect.width, page.rect.height
        print(f"页面尺寸: {width} x {height}")


if __name__ == "__main__":
    # load_data_with_unstructured(data_dir)
    # print("________unstructured end___________\n")
    # load_data_with_langchain_loader(data_dir)
    # print("________langchin end___________\n")
    # load_data_with_pymupdf(data_dir)
    # print("________pymupdf end___________\n")
    load_data_with_llama_index(data_dir)
    print("________llama index end___________\n")