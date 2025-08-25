import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pymilvus import MilvusClient
import pymilvus.model as pymilvus_model
from glob import glob
from tqdm import tqdm
import json
from unstructured.partition.pdf import partition_pdf

load_dotenv(verbose=True)
collection_name = "mfd_rag_collection"

def init_data():
    milvus_client = MilvusClient(uri="./milvus_demo.db")
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name=collection_name)
    embedding_model = pymilvus_model.DefaultEmbeddingFunction()
    test_embedding = embedding_model.encode_queries(["This is a test"])[0]
    embedding_dim = len(test_embedding)
    milvus_client.create_collection(
        collection_name=collection_name,
        metric_type="IP",  # 内积距离
        consistency_level="Strong",
        dimension=embedding_dim,
        # 支持的值为 (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`)。更多详情请参见 https://milvus.io/docs/consistency.md#Consistency-Level。
    )

    docs = partition_pdf(
        filename="/Users/yegaosong/develop/ai-learn/rag-practise/data/content-chunking.pdf")
    parsed_content = []
    for elem in docs:
        print("element.text", elem.text)
        # 过滤无效内容（如空白）
        if hasattr(elem, "text") and elem.text.strip():
            parsed_content.append({
                "text": elem.text,
                "metadata": {
                    "page_number": elem.metadata.page_number,  # 页码（关键元数据）
                    "content_type": elem.__class__.__name__  # 内容类型（如TextBlock、Table）
                }
            })
    texts = [item["text"] for item in parsed_content]  # 提取纯文本列表

    doc_embeddings = embedding_model.encode_documents(texts)
    insert_docs = []
    for i, line in enumerate(tqdm(texts, desc="Creating embeddings")):
        insert_docs.append({"id": i, "vector": doc_embeddings[i], "text": line})

    milvus_client.insert(collection_name=collection_name, data=insert_docs)

def search_db(question: str):
    milvus_client = MilvusClient(uri="./milvus_demo.db")
    embedding_model = pymilvus_model.DefaultEmbeddingFunction()
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=embedding_model.encode_queries(
            [question],
        ),  # 将问题转换为嵌入向量
        limit=100,  # 返回前3个结果
        search_params={"metric_type": "IP", "params": {}},  # 内积距离
        output_fields=["text"],  # 返回 text 字段
    )
    return search_res
def main():
    question = '将上下文关键信息以markdown 格式化输出'
    init_data()
    search_res = search_db(question)

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    print(context)

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
    SYSTEM_PROMPT = """
       Human: 你是一个 AI 助手。你能够从提供的上下文段落片段中找到问题的答案。
       """
    USER_PROMPT = f"""
       请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。最后追加原始回答的中文翻译，并用 <translated>和</translated> 标签标注。
       <context>
       {context}
       </context>
       <question>
       {question}
       </question>
       <translated>
       </translated>
       """
    # 使用强类型消息参数
    messages = [
        ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=USER_PROMPT),
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        stream=False
    )
    print(response.choices[0].message.content)

if __name__ == '__main__':
    main()