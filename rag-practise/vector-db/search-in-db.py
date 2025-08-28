
from pydoc import cli
from pymilvus import MilvusClient, DataType
import random
# 1. 设置 Milvus 客户端
client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "ann_search_demo"
def init_db(metric_type):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    # 2. 创建 schema
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
    schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="docId", datatype=DataType.INT64)

    # 3. 创建集合
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type=metric_type,
        index_type="FLAT",
        index_name="vector_index",
        params={}
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
        sync=True
    )
def init_data():
    num_vectors = 1000
    vectors = [[random.random() for _ in range(128)] for _ in range(num_vectors)]
    ids = list(range(num_vectors))
    colors = [f"color_{random.randint(1, 1000)}" for _ in range(num_vectors)]
    doc_ids = [random.randint(1, 100) for _ in range(num_vectors)]  # 假设有100个文档
    entities = [{"id": ids[i], "vector": vectors[i], "docId": doc_ids[i], "color": colors[i]} for i in range(num_vectors)]
    client.insert(collection_name=COLLECTION_NAME, data=entities)

    client.load_collection(collection_name=COLLECTION_NAME)

def vector_search():
    query_vectors = [[random.random() for _ in range(128)] for _ in range(2)]
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vectors,
        anns_field="vector",
        limit=3,
        search_params={"metric_type": "L2"},
        output_fields=["color"]
    )

    print("批量搜索结果:")
    for i, hits in enumerate(results):
        print(f"\n查询向量 {i+1} 的结果:")
        for hit in hits:
            print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# def search_by_text():
def search_filter():
    query_vector = [random.random() for _ in range(128)]
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        limit=3,
        search_params={"metric_type": "L2"},
        filter='color like "color_%"',  # 过滤条件：颜色以color_开头且点赞数大于500
        output_fields=["color"]  # 指定输出字段
    )
    for hits in results:
        for hit in hits:
            print(f"ID: {hit['id']}, 距离: {hit['distance']}, 颜色: {hit['entity']['color']}")
# def query_from_db():
def group_search():
    query_vector = [random.random() for _ in range(128)]
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        limit=5,  # 返回5个不同的文档组
        group_by_field="docId",  # 按文档ID分组
        output_fields=["docId", "color"]
    )
    for hits in results:
        for hit in hits:
            print(f"ID: {hit['id']}, 距离: {hit['distance']}, 颜色: {hit['entity']['color']}")
def range_search():
    query_vectors = [[random.random() for _ in range(128)] for _ in range(2)]

    # 使用 L2 距离度量，设置范围搜索参数
    # 注意：对于 L2 距离，range_filter 应该大于 radius
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vectors,
        anns_field="vector",
        limit=10,  # 增加限制以显示更多结果
        search_params={
            "metric_type": "IP",
            "params": {
                "radius": 1.0,  # 外圈半径
                "range_filter": 1.5 # "L2" 不支持该参数
            }
        },
        output_fields=["color"]
    )    
def search_by_iterator(): 
    query_vector = [random.random() for _ in range(128)]

    # 创建 SearchIterator
    iterator = client.search_iterator(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        search_params={"metric_type": "L2"},
        batch_size=1000,  # 每批返回1000条结果
        limit=20000,      # 总共返回20000条结果
        output_fields=["color"]
    )
    all_results = []
    while True:
        result = iterator.next()
        if not result:
            iterator.close()
            break
        
        # 将结果转换为字典并添加到结果列表
        for hit in result:
            all_results.append(hit.to_dict())

    print(f"总共获取到 {len(all_results)} 条结果")
    print("\n前5条结果:")
    for result in all_results[:5]:
        print(f"ID: {result['id']}, 距离: {result['distance']}, 颜色: {result['entity']['color']}")

if __name__ == "__main__":
    metric_type = "L2"  
    # metric_type = "IP"  # range search 时必须使用 内积
    init_db(metric_type)
    init_data()
    # vector_search()
    # search_filter()
    # range_search()
    # group_search()
    search_by_iterator()
    client.release_collection(collection_name=COLLECTION_NAME)
