import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pymilvus import MilvusClient
import pymilvus.model as pymilvus_model
from glob import glob
from tqdm import tqdm
import json

load_dotenv(verbose=True)

def main():
    milvus_client = MilvusClient(uri="./milvus_demo.db")

    collection_name = "my_rag_collection"
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    embedding_model = pymilvus_model.DefaultEmbeddingFunction()
    test_embedding = embedding_model.encode_queries(["This is a test"])[0]
    embedding_dim = len(test_embedding)

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",  # 内积距离
        consistency_level="Strong")

    text_lines = []

    for file_path in glob("./docs/*.md", recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()

        text_lines += file_text.split("# ")

    print(text_lines)

    data = []
    doc_embeddings = embedding_model.encode_documents(text_lines)

    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": doc_embeddings[i], "text": line})
    milvus_client.insert(collection_name=collection_name, data=data)
    question = "How is data stored in milvus?"
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=embedding_model.encode_queries(
            [question],
        ),  # 将问题转换为嵌入向量
        limit=3,  # 返回前3个结果
        search_params={"metric_type": "IP", "params": {}},  # 内积距离
        output_fields=["text"],  # 返回 text 字段
    )
    print(search_res)
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    print(json.dumps(retrieved_lines_with_distances, indent=4))
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )
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