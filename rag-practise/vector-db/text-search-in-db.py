
from pydoc import cli
from pymilvus import MilvusClient, DataType, Function, FunctionType
import json
# 1. 设置 Milvus 客户端
client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "text_search_demo"
def init_db():
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )

    # 将函数添加到 schema
    schema.add_function(bm25_function)

    # 4. 配置索引参数
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        },
    )

    # 5. 创建集合
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

# 6. 插入文本数据
def init_data():
    documents = [
        {'text': '信息检索是一个研究领域。'},
        {'text': "信息检索"},
        {'text': '信息检索专注于在大型数据集中查找相关信息。'},
        {'text': '数据挖掘和信息检索在研究中有所重叠。'},
        {'text': '搜索引擎是信息检索系统的一个典型例子。'},
        {'text': '自然语言处理在信息检索中扮演重要角色。'},
    ]
    result= client.insert(COLLECTION_NAME, documents)
    print(result)

    client.load_collection(collection_name=COLLECTION_NAME)

def full_text_search():
    search_params = {
        'params': {'drop_ratio_search': 0.1},
    }

    query_text = "信息"
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_text],
        anns_field='sparse',
        limit=3,
        search_params=search_params,
        output_fields=["text"]  # 添加输出字段以显示原始文本
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n搜索结果:")
    if results and len(results) > 0:
        for hits in results:
            for hit in hits:
                # 打印完整的 hit 结构
                print("\nHit 结构:")
                print(json.dumps(hit, indent=2, ensure_ascii=False))
                # 尝试不同的字段访问方式
                if 'entity' in hit:
                    print(f"ID: {hit.get('id', 'N/A')}, 文本: {hit['entity'].get('text', 'N/A')}")
                else:
                    print("未找到 entity 字段")
    else:
        print("没有找到匹配的结果")


if __name__ == "__main__":
    init_db()
    init_data()
    # group_search()
    full_text_search()
    client.release_collection(collection_name=COLLECTION_NAME)
