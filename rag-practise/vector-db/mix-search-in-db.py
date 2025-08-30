
from pydoc import cli
from pymilvus import MilvusClient, DataType, FieldSchema, FieldSchema, CollectionSchema, Collection
from milvus_model.hybrid import BGEM3EmbeddingFunction
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import json
import time
import os
# 1. 设置 Milvus 客户端
client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "mix_search_demo"
DEVICE = "cpu" # 或者 "cuda" 如果有GPU并已正确配置
BATCH_SIZE = 50 # 可以尝试减小批次大小，例如 10 或 20，进行测试

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data')

model_name = "BAAI/bge-large-zh-v1.5"  # 中文优化版本
model_kwargs = {'device': 'cpu'}  # 若有GPU可改为 'cuda'
encode_kwargs = {'normalize_embeddings': True}  # 归一化向量，便于计算余弦相似度

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)



def init_db():
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
    ]
    schema = CollectionSchema(fields, description="Wukong Hybrid Search Collection v4")

    print(f"正在创建集合 '{COLLECTION_NAME}'...")

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP"
    )

    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="IP"
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print(f"集合 '{COLLECTION_NAME}' 创建成功。")

    print(f"正在加载集合 '{COLLECTION_NAME}'...")
    client.load_collection(COLLECTION_NAME)

# 6. 插入文本数据
def init_data():
    DATA_PATH = f"{data_dir}/role.json"
   # 1. 加载数据
    print(f"1. 正在从 {DATA_PATH} 加载数据...")
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据文件 {DATA_PATH} 未找到。请检查路径。")
        exit()
    except json.JSONDecodeError:
        print(f"错误: 数据文件 {DATA_PATH} JSON 格式错误。")
        exit()

    docs = []
    metadata = []
    for item in dataset.get('data', []): # 使用 .get 避免 'data' 键不存在的错误
        text_parts = [item.get('title', ''), item.get('description', '')]
        if 'combat_details' in item and isinstance(item['combat_details'], dict):
            text_parts.extend(item['combat_details'].get('combat_style', []))
            text_parts.extend(item['combat_details'].get('abilities_used', []))
        if 'scene_info' in item and isinstance(item['scene_info'], dict):
            text_parts.extend([
                item['scene_info'].get('location', ''),
                item['scene_info'].get('environment', ''),
                item['scene_info'].get('time_of_day', '')
            ])
        # 过滤掉 None 和空字符串，然后连接
        docs.append(' '.join(filter(None, [str(part).strip() for part in text_parts if part])))
        metadata.append(item)

    if not docs:
        print("错误: 未能从数据文件中加载任何文档。请检查文件内容和结构。")
        exit()
    print(f"数据加载完成，共 {len(docs)} 条文档。")

    # 2. 生成向量
    print("2. 正在生成向量...")
    try:
        docs_to_embed = docs
        print(f"将为 {len(docs_to_embed)} 条文档生成向量...")
        docs_embeddings = embeddings.aembed_documents(docs_to_embed)
        print("向量生成完成。")
        if "sparse" in docs_embeddings and docs_embeddings["sparse"].shape[0] > 0:
            print(f"  稀疏向量类型 (整体): {type(docs_embeddings['sparse'])}")
            #  打印第一个稀疏向量的形状和部分内容以供检查
            first_sparse_vector_row_obj = docs_embeddings['sparse'][0] # 这会得到一个表示单行的稀疏数组对象
            print(f"  第一个稀疏向量 (行对象类型): {type(first_sparse_vector_row_obj)}")
            print(f"  第一个稀疏向量 (行对象形状): {first_sparse_vector_row_obj.shape}")
            if hasattr(first_sparse_vector_row_obj, 'col') and hasattr(first_sparse_vector_row_obj, 'data'):
                print(f"  第一个稀疏向量 (部分列索引/col): {first_sparse_vector_row_obj.col[:5]}")
                print(f"  第一个稀疏向量 (部分数据/data): {first_sparse_vector_row_obj.data[:5]}")
            elif hasattr(first_sparse_vector_row_obj, 'indices') and hasattr(first_sparse_vector_row_obj, 'data'): # Fallback for other types
                print(f"  第一个稀疏向量 (部分索引/indices): {first_sparse_vector_row_obj.indices[:5]}")
                print(f"  第一个稀疏向量 (部分数据/data): {first_sparse_vector_row_obj.data[:5]}")
            else:
                print("  无法直接获取第一个稀疏向量的列索引和数据属性。")
        else:
            print("警告: 未生成稀疏向量或稀疏向量为空。")

    except Exception as e:
        print(f"生成向量时发生错误: {e}")
        exit()






if __name__ == "__main__":
    init_db()
    init_data()
    # group_search()
