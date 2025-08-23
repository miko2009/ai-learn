from pydoc import doc
from langchain_community.document_loaders import WebBaseLoader
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from llama_index.readers.web import SimpleWebPageReader
import requests



def load_data_with_llama_index(page_url):
    simple_file_loader = SimpleWebPageReader(html_to_text=True)
    docs = simple_file_loader.load_data([page_url])
    print("llama index docs: ", docs)
    # 通过vars函数查看所有可用的元数据
    for i, element in enumerate(docs):
        print(f"\n--- Element {i+1} ---")
        print(f"类型: {type(element)}")

    return docs


def load_data_with_langchain_loader(page_url):
    loader = WebBaseLoader(web_paths=[page_url])
    docs = loader.load()
    print("langchain docs: ", docs)
    # 通过vars函数查看所有可用的元数据
    for i, element in enumerate(docs):
        print(f"\n--- Element {i+1} ---")
        print(f"类型: {type(element)}")

    return docs


def load_data_with_unstructured(page_url):
    try:
        # 发送请求获取网页内容

        response = requests.get(page_url)
        response.raise_for_status()  # 检查请求是否成功

        # 获取HTML内容
        html_content = response.text
        # 使用unstructured解析HTML
        elements = partition_html(text=html_content)
        
        # 按标题分块（可选，根据需要处理）
        chunks = chunk_by_title(elements)
        print(chunks)
        return chunks
    
    except requests.exceptions.RequestException as e:
        print(f"请求网页失败: {e}")
        return None
    except Exception as e:
        print(f"解析网页内容失败: {e}")
        return None
    return docs


if __name__ == "__main__":
    page_url = "https://zh.wikipedia.org/wiki/黑神话：悟空"
    # load_data_with_unstructured(page_url)
    # print("________unstructured end___________\n")
    # load_data_with_langchain_loader(page_url)
    # print("________langchain end___________\n")
    load_data_with_llama_index(page_url)
    print("________llama index end___________\n")