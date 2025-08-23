from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
from llama_index.core import SimpleDirectoryReader

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data')


def load_data_with_llama_index_md(data_dir):
    simple_file_loader = SimpleDirectoryReader(input_dir=data_dir, required_exts=['.md'])
    docs_simple_file = simple_file_loader.load_data()
    print("Simple MD file with LLama Index loader: ", docs_simple_file)
    return docs_simple_file


def load_data_with_llama_index(data_dir):
    loader = SimpleDirectoryReader(input_dir=data_dir)
    docs = loader.load_data()
    print("Simple directory loader: ", len(docs))

    simple_file_loader = SimpleDirectoryReader(input_files=[os.path.join(data_dir, 'simple.txt')])
    docs_simple_file = simple_file_loader.load_data()
    print("Simple file loader: ", len(docs_simple_file))
    return docs

def load_data_all(data_dir):
    loader = DirectoryLoader(data_dir, 
                         glob="**/*",
                         )
    docs = loader.load()
    return docs

def load_data_md(data_dir):    
    loader = DirectoryLoader(data_dir, 
                         glob="**/*.md", 
                         use_multithreading=True,
                         show_progress=True,
                         )
    docs = loader.load()
    print("MD docs with directory loader: ", docs)
    return docs

def load_data_with_specific_tool(data_dir):
    loader = DirectoryLoader(data_dir, 
                         glob="**/*.txt", 
                         use_multithreading=True,
                         loader_cls=TextLoader
                         )
    docs = loader.load()
    return docs

def main():
    # docs_all = load_data_all(data_dir)
    # print("All docs: ", len(docs_all))
    # docs_with_tools = load_data_with_specific_tool(data_dir)
    # print("Docs with tools: ", len(docs_with_tools))
    docs_md = load_data_md(data_dir)
    print("Docs md: ", len(docs_md))
    # docs_simple_directory = load_data_with_llama_index(data_dir)
    # print("Docs simple directory: ", len(docs_simple_directory))
    docs_simple_directory_md = load_data_with_llama_index_md(data_dir)
    print("Docs simple directory MD: ", len(docs_simple_directory_md))

if __name__ == "__main__":
    main()
    exit()