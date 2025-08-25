from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
collection_name = "quick_setup"
if collection_name in client.list_collections():
    client.drop_collection(collection_name=collection_name)
    print(f"✓ 已删除已存在的集合 {collection_name}")

# 创建新集合
client.create_collection(
    collection_name=collection_name,
    dimension=5
)
print(f"✓ {collection_name} 已创建")

cols = client.list_collections()
info = client.describe_collection(collection_name=collection_name)
print(f"{collection_name} 详情：", info)

new_collection_name = "quick_renamed"
if new_collection_name in client.list_collections():
    client.drop_collection(collection_name=new_collection_name)
    print(f"✓ 已删除已存在的集合 {new_collection_name}")

client.rename_collection(
    old_name=collection_name,
    new_name=new_collection_name
)
print(f"✓ {collection_name} 已重命名为 {new_collection_name}")

## “Time-To-Live”“存活时间”，表示数据在集合中可保留的最大时长；
client.alter_collection_properties(
    collection_name=new_collection_name,
    properties={"collection.ttl.seconds": 60}
)
print(f"✓ 已为 {new_collection_name} 设置 TTL=60s")

client.drop_collection_properties(
    collection_name=new_collection_name,
    property_keys=["collection.ttl.seconds"]
)
print(f"✓ 已删除 {new_collection_name} 的 TTL 属性")

client.load_collection(collection_name=new_collection_name)
state = client.get_load_state(collection_name=new_collection_name)
print("加载状态：", state)

client.release_collection(collection_name=new_collection_name)
state = client.get_load_state(collection_name=new_collection_name)
print("释放后状态：", state)

client.create_partition(
    collection_name=new_collection_name,
    partition_name="partA"
)
print("✓ 已创建 partition partA")
print("更新后 Partition 列表：", client.list_partitions(new_collection_name))

exists = client.has_partition(
    collection_name=new_collection_name,
    partition_name="partA"
)
print("partA 存在？", exists)

client.load_partitions(
    collection_name=new_collection_name,
    partition_names=["partA"]
)
print("partA 加载状态：", client.get_load_state(new_collection_name, partition_name="partA"))

client.release_partitions(
    collection_name=new_collection_name,
    partition_names=["partA"]
)
print("partA 释放后状态：", client.get_load_state(new_collection_name, partition_name="partA"))

client.drop_partition(
    collection_name=new_collection_name,
    partition_name="partA"
)
print("✓ 已删除 partition partA")
print("最终 Partition 列表：", client.list_partitions(new_collection_name))

client.create_alias(collection_name=new_collection_name, alias="alias3")
client.create_alias(collection_name=new_collection_name, alias="alias4")
print("✓ 已创建 alias3, alias4")

aliases = client.list_aliases(collection_name=new_collection_name)
print("当前 aliases：", aliases)

desc = client.describe_alias(alias="alias3")
print("alias3 详情：", desc)

client.alter_alias(collection_name=new_collection_name, alias="alias4")
print("✓ 已将 alias4 重新分配给 quick_renamed")

client.drop_alias(alias="alias3")
print("剩余 aliases：", client.list_aliases(new_collection_name))

client.drop_collection(collection_name=new_collection_name)
print(f"✓ 集合 {new_collection_name} 已删除")