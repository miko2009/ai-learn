import os

from urllib3 import response
from  base import ImageOCRReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
import dotenv
dotenv.load_dotenv()

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

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

def main():

    # imageOcrReader = ImageOCRReader(
    #     lang='ch',
    #     use_gpu= False
    # )
    # docs = imageOcrReader.load_data(f"{current_dir}/rag-state.png")
    # from llama_index.core import VectorStoreIndex
    # index = VectorStoreIndex.from_documents(docs)
    # query_engine = index.as_query_engine()

    # response = query_engine.query("检索召回下一步是什么,在第几个文本块, 总共有几个文本块")
    # print(response)

    # imageOcrReader = ImageOCRReader(
    #     lang='ch',
    #     use_gpu= False
    # )
    # docs = imageOcrReader.load_data(f"{current_dir}/rag.png")
    # from llama_index.core import VectorStoreIndex
    # index = VectorStoreIndex.from_documents(docs)
    # query_engine = index.as_query_engine()

    # response = query_engine.query("页码是多少")
    # print(response)



    files = [
        # f"{current_dir}/data/complex.png",
        # f"{current_dir}/data/car.png",
        # f"{current_dir}/data/rag.png",
        # f"{current_dir}/data/rag-state.png",
        #f"{current_dir}/data/repeat.png",
        # f"{current_dir}/data/num.png",
        f"{current_dir}/data/en.png"
    ]
    imageOcrReader = ImageOCRReader(
        lang='ch',
        use_gpu= False
    )
    docs = imageOcrReader.load_data(files)
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,  # 返回top 3最相关的文档
        return_score=True    # 关键：启用返回分数
    )

    # 4. 执行检索（获取带分数的文档）
    query = "沪开头的车牌号"
    nodes_with_score = retriever.retrieve(query)
    # 这个问题答案是错误. 改成: 沪开头的车牌号
    # response = retriever.retrieve("沪的车牌号")
    # response = retriever.retrieve("沪开头的车牌号")
    # --- car image 

    # response = retriever.retrieve("放松减压方式")
    # response = retriever.retrieve("签名日期是多少")
    # --- complex imaged
    # response = retriever.retrieve("检索召回下一步是什么,在第几个文本块, 总共有几个文本块")
    # --- rag-state image
    # response = retriever.retrieve("页码是多少")
    # ---rag imageo
    # response = retriever.retrieve("列出数字")
    # ---- repeate image
    print(nodes_with_score)
    for i, node in enumerate(nodes_with_score, 1):
        print(node)
        print(f"[Text Block {node.metadata}] (conf: {node.score}) ")

 
    
        
# 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
# 在根目录可以通过python -m ocr_research.main 运行
    pass

if __name__ == "__main__":
    main()