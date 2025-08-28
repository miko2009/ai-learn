# text2sql_query.py
import os
import logging
import yaml
import re
from dotenv import load_dotenv
from pymilvus import MilvusClient
from pymilvus import model, CollectionSchema, FieldSchema, DataType
from sqlalchemy import create_engine, text
import pymysql
import json
from openai import OpenAI

user = "root"
password = "password"
db_name = "sakila"


# 1. 环境与日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # 加载 .env 环境变量

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 


client = MilvusClient(uri="./text2sql_milvus_sakila.db")

# 6. 数据库连接（SAKILA）
DB_URL = os.getenv(
    "SAKILA_DB_URL", 
    "mysql+pymysql://root:password@localhost:3306/sakila"
)
engine = create_engine(DB_URL)

def init_embedding():
    return model.DefaultEmbeddingFunction()

def store_hypothetical_to_vector():
    with open(f"{data_dir}/q2sql_pairs.json", "r") as f:
        pairs = json.load(f)
    logging.info(f"[Q2SQL] 从JSON文件加载了 {len(pairs)} 个问答对")
    embedding_function = init_embedding()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="sql_text", dtype=DataType.VARCHAR, max_length=2000),
    ]
    schema = CollectionSchema(fields, description="Q2SQL Knowledge Base", enable_dynamic_field=False)

    # 5. 创建 Collection（如不存在）
    collection_name = "q2sql_knowledge"
    if not client.has_collection(collection_name):
        client.create_collection(collection_name=collection_name, schema=schema)
        logging.info(f"[Q2SQL] 创建了新的集合 {collection_name}")
    else:
        logging.info(f"[Q2SQL] 集合 {collection_name} 已存在")

    # 6. 为向量字段添加索引
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE", params={"nlist": 1024})
    client.create_index(collection_name=collection_name, index_params=index_params)

    # 7. 批量插入 Q2SQL 对
    data = []
    texts = []
    for pair in pairs:
        texts.append(pair["question"])
        data.append({"question": pair["question"], "sql_text": pair["sql"]})

    logging.info(f"[Q2SQL] 准备处理 {len(data)} 个问答对")

    # 生成全部嵌入
    embeddings = embedding_function(texts)
    logging.info(f"[Q2SQL] 成功生成了 {len(embeddings)} 个向量嵌入")

    # 组织为 Milvus insert 格式
    records = []
    for emb, rec in zip(embeddings, data):
        rec["vector"] = emb
        records.append(rec)

    res = client.insert(collection_name=collection_name, data=records)
    logging.info(f"[Q2SQL] 成功插入了 {len(records)} 条记录到Milvus")
    logging.info(f"[Q2SQL] 插入结果: {res}")

    logging.info("[Q2SQL] 知识库构建完成")

def store_ddl_to_vector(ddl_map):
    #    字段：id, vector, table_name, ddl_text
    embedding_fn = init_embedding()
    vector_dim = 768
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="ddl_text", dtype=DataType.VARCHAR, max_length=2000),
    ]
    schema = CollectionSchema(fields, description="DDL Knowledge Base", enable_dynamic_field=False)
    collection_name = "ddl_knowledge"
    if not client.has_collection(collection_name):
        client.create_collection(collection_name=collection_name, schema=schema)
        logging.info(f"[DDL] 创建了新的集合 {collection_name}")
    else:
        logging.info(f"[DDL] 集合 {collection_name} 已存在")

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE", params={"nlist": 1024})
    client.create_index(collection_name=collection_name, index_params=index_params)

    # 7. 批量插入 DDL
    data = []
    texts = []
    for tbl, ddl in ddl_map.items():
        texts.append(ddl)
        data.append({"table_name": tbl, "ddl_text": ddl})

    logging.info(f"[DDL] 准备处理 {len(data)} 个表/视图的DDL语句")

    # 生成全部嵌入
    embeddings = embedding_fn(texts)
    logging.info(f"[DDL] 成功生成了 {len(embeddings)} 个向量嵌入")

    # 组织为 Milvus insert 格式
    records = []
    for emb, rec in zip(embeddings, data):
        rec["vector"] = emb
        records.append(rec)

    res = client.insert(collection_name=collection_name, data=records)


def generate_ddl_file(): 
    conn = pymysql.connect(
        host="localhost", port=3306, user=user, password=password,
        database=db_name, cursorclass=pymysql.cursors.Cursor
    )
    ddl_map = {}
    try:
        with conn.cursor() as cursor:
            # 3. 获取所有表名
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s;", (db_name,)
            )  
            tables = [row[0] for row in cursor.fetchall()]

            # 4. 遍历表列表，执行 SHOW CREATE TABLE
            for tbl in tables:
                cursor.execute(f"SHOW CREATE TABLE `{db_name}`.`{tbl}`;")
                result = cursor.fetchone()
                # result[0]=表名, result[1]=完整 DDL
                ddl_map[tbl] = result[1]  

    finally:
        conn.close()
    with open(f"{data_dir}/dl_statements.yaml", "w") as f:
        yaml.safe_dump(ddl_map, f, sort_keys=True, allow_unicode=True)
    return ddl_map
# 7. 检索函数
def retrieve(collection: str, query_emb: list, top_k: int = 3, fields: list = None):
    results = client.search(
        collection_name=collection,
        data=[query_emb],
        limit=top_k,
        output_fields=fields
    )
    logging.info(f"[检索] {collection} 检索结果: {results}")
    return results[0]  # 返回第一个查询的结果列表

# 8. SQL 提取函数
def extract_sql(text: str) -> str:
    # 尝试匹配 SQL 代码块
    sql_blocks = re.findall(r'```sql\n(.*?)\n```', text, re.DOTALL)
    if sql_blocks:
        return sql_blocks[0].strip()
    
    # 如果没有找到代码块，尝试匹配 SELECT 语句
    select_match = re.search(r'SELECT.*?;', text, re.DOTALL)
    if select_match:
        return select_match.group(0).strip()
    
    # 如果都没有找到，返回原始文本
    return text.strip()

# 9. 核心流程：自然语言 -> SQL -> 执行 -> 返回
def text2sql(question: str):
    embedding_fn = init_embedding()
    # 9.1 用户提问嵌入
    q_emb = embedding_fn([question])[0]
    logging.info(f"[检索] 问题嵌入完成")

    # 9.2 RAG 检索：DDL
    ddl_hits = retrieve("ddl_knowledge", q_emb.tolist(), top_k=3, fields=["ddl_text"])
    logging.info(f"[检索] DDL检索结果: {ddl_hits}")
    try:
        ddl_context = "\n".join(hit.get("ddl_text", "") for hit in ddl_hits)
    except Exception as e:
        logging.error(f"[检索] DDL处理错误: {e}")
        ddl_context = ""

    # 9.3 RAG 检索：示例对
    q2sql_hits = retrieve("q2sql_knowledge", q_emb.tolist(), top_k=3, fields=["question", "sql_text"])
    logging.info(f"[检索] Q2SQL检索结果: {q2sql_hits}")
    try:
        example_context = "\n".join(
            f"NL: \"{hit.get('question', '')}\"\nSQL: \"{hit.get('sql_text', '')}\"" 
            for hit in q2sql_hits
        )
    except Exception as e:
        logging.error(f"[检索] Q2SQL处理错误: {e}")
        example_context = ""

    # # 9.4 RAG 检索：字段描述
    # desc_hits = retrieve("dbdesc_knowledge", q_emb.tolist(), top_k=8, fields=["table_name", "column_name", "description"])
    # logging.info(f"[检索] 字段描述检索结果: {desc_hits}")
    # try:
    #     desc_context = "\n".join(
    #         f"{hit.get('table_name', '')}.{hit.get('column_name', '')}: {hit.get('description', '')}"
    #         for hit in desc_hits
    #     )
    # except Exception as e:
    #     logging.error(f"[检索] 字段描述处理错误: {e}")
    #     desc_context = ""

    # 9.5 Prompt 组装
    prompt = (
        f"### Schema Definitions:\n{ddl_context}\n"
        f"### Examples:\n{example_context}\n"
        f"### Query:\n\"{question}\"\n"
        "请只返回SQL语句，不要包含任何解释或说明。"
    )
    logging.info("[生成] 开始生成SQL")
    client = OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个SQL专家。请只返回SQL查询语句，不要包含任何Markdown格式或其他说明。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    raw_sql = response.choices[0].message.content.strip()
    sql = extract_sql(raw_sql)
    logging.info(f"[生成] 原始输出: {raw_sql}")
    logging.info(f"[生成] 提取的SQL: {sql}")

    # 9.7 执行并打印结果
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            cols = result.keys()
            rows = result.fetchall()
            print("\n查询结果：")
            print("列名：", cols)
            for r in rows:
                print(r)
    except Exception as e:
        logging.error(f"[执行] 执行失败: {e}")
        print("执行错误：", e)

# 10. 程序入口
if __name__ == "__main__":
    user_q = input("请输入您的自然语言查询： ")
    ddl_map = generate_ddl_file()
    #store_ddl_to_vector(ddl_map)
    # store_hypothetical_to_vector()
    text2sql(user_q)