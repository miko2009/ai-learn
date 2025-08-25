from math import sin
from pydoc import doc
from langchain_community.document_loaders import PyPDFLoader
from numpy import single
from unstructured.partition.pdf import partition_pdf
from llama_index.core import SimpleDirectoryReader
import os
import pymupdf
from llama_parse import LlamaParse
from dotenv import load_dotenv

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 

load_dotenv()
llama_key = os.getenv("LLAMA_KEY")
def load_data_with_llama_index(single_file):
    docs = LlamaParse(api_key = llama_key, result_type="markdown",   preserve_layout_alignment_across_pages=True).load_data(single_file)
    for i, doc in enumerate(docs, 1):
        print(doc.text)
    return docs


def load_data_with_langchain_loader(single_file):
    loader = PyPDFLoader(single_file)
    docs = loader.load()
    print("langchain docs: ", docs)
    # 通过vars函数查看所有可用的元数据
    for i, element in enumerate(docs):
        print(f"\n--- Element {i+1} ---")
        print(f"类型: {element}")

    return docs


def load_data_with_unstructured(single_file):
    elements = partition_pdf(filename=single_file, strategy="hi_res", infer_table_structure=True)
    print("Unstructured docs: ", elements)
    element_map = {element.id: element for element in elements if hasattr(element, 'id')}

    for element in elements:
        print("category", element.category)
        if element.category == "Table":
            print("\n表格数据:")
            print("表格元数据:", vars(element.metadata))  # 使用vars()显示所有元数据属性
            print("表格内容:")
            print(element.text)  # 打印表格文本内容
            
            # 获取并打印父节点信息
            parent_id = getattr(element.metadata, 'parent_id', None)
            if parent_id and parent_id in element_map:
                parent_element = element_map[parent_id]
                print("\n父节点信息:")
                print(f"类型: {parent_element.category}")
                print(f"内容: {parent_element.text}")
                if hasattr(parent_element, 'metadata'):
                    print(f"父节点元数据: {vars(parent_element.metadata)}")  # 同样使用vars()显示所有元数据
            else:
                print(f"未找到父节点 (ID: {parent_id})")
            print("-" * 50)

    text_elements = [el for el in elements if el.category == "Text"]
    table_elements = [el for el in elements if el.category == "Table"]
    return elements

def load_data_with_pymupdf(single_file): 
    doc = pymupdf.open(single_file)
    print("=== PyMuPDF 基本信息提取 ===")
    print(f"文档页数: {doc}")
    print(f"文档标题: {doc.metadata['title']}")
    print(f"文档作者: {doc.metadata['author']}")
    print(f"文档元数据: {doc.metadata}")  # 比Unstructured提供更多元数据
  
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
    single_file = f"{data_dir}/billionaires_page-1-5.pdf"
    # load_data_with_unstructured(single_file)
    print("________unstructured end___________\n")
    # load_data_with_langchain_loader(single_file)
    print("________langchin end___________\n")
    # load_data_with_pymupdf(single_file)
    print("________ pymupdf end___________\n")
    load_data_with_llama_index(single_file)
    print("________llama index end___________\n")