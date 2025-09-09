
from ast import main
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from glob import glob
from tqdm import tqdm
import json
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from typing import Dict, Any, List
from dataclasses import dataclass
from neo4j import GraphDatabase

import dotenv

dotenv.load_dotenv()

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
@dataclass
class Entity:
    """
    实体数据类
    
    用于存储从文本中提取的实体信息，包括：
    - name: 实体名称（如 "Paul Atreides"）
    - type: 实体类型（如 "Person", "House", "Planet"）
    - properties: 实体的额外属性（可选）
    """
    name: str                           # 实体名称，必填
    type: str                           # 实体类型，必填
    properties: Dict[str, Any] = None   # 实体属性，可选
    
    def __post_init__(self):
        """初始化后处理，确保 properties 不为 None"""
        if self.properties is None:
            self.properties = {}
@dataclass
class Relationship:
    """
    关系数据类
    
    用于存储实体间的关系信息，包括：
    - source: 源实体名称
    - target: 目标实体名称  
    - type: 关系类型（如 "PARENT_OF", "HEIR_OF"）
    - properties: 关系的额外属性（可选）
    """
    source: str                         # 源实体名称
    target: str                         # 目标实体名称
    type: str                           # 关系类型
    properties: Dict[str, Any] = None   # 关系属性，可选
    
    def __post_init__(self):
        """初始化后处理，确保 properties 不为 None"""
        if self.properties is None:
            self.properties = {}


class ExtractEntities:
    def __init__(self, llm) -> None:
        self.llm = llm
    def get_entity_from_question(self, question):
        prompt =f""" 
        你是一个实体关系问题专家，善于从问题中提取实体名称, 并能判断问题是否需要多级遍历
        文本：{question}
        
        返回JSON格式：
        {{
            name: "实体名称", need_loop: "bool值是否需要多级遍历"
        }}
        """
        response = self.llm.complete(prompt)
        questionDict = json.loads(response.text)
        return questionDict
    def get_entities_by_llm(self, text: str):
        prompt = f"""
            从文本中提取公司实体：
        
            文本：{text}
        
            返回格式如下的JSON：
            {{
                "entities": [
                    {{"name": "公司名", "type": "Company"}}
                ]
            }}
            并可直接使用 json.loads 变成dict
            """
        response = self.llm.complete(prompt)
        entitiesDict = json.loads(response.text)
        entities = [Entity(e["name"], e["type"]) for e in entitiesDict.get("entities", [])]
        print(f" 提取到 {len(entities)} 个公司实体")
        return entities
    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
    
        """步骤2: 提取控股关系"""
        entity_names = [e.name for e in entities]
        
        prompt = f"""
        从文本中提取公司间的控股关系：
        
        文本：{text}
        公司：{entity_names}
        
        返回JSON格式：
        {{
            "relationships": [
                {{"source": "母公司", "target": "子公司", "type": "CONTROLS"}}
            ]
        }}
        """
        
        response = self.llm.complete(prompt)
        result = json.loads(response.text)
        
        relationships = []
        for r in result.get("relationships", []):
            if r["source"] in entity_names and r["target"] in entity_names:
                relationships.append(Relationship(r["source"], r["target"], r["type"]))
        
        print(f" 提取到 {len(relationships)} 个控股关系")
        return relationships
class DbOperation:
    def __init__(self, driver) -> None:
        self.driver = driver
    def store_entities_to_db(self, entities, relationships):
         with self.driver.session() as session:
            # 清空现有数据
            session.run("MATCH (n) DETACH DELETE n")
            
            # 写入公司实体
            for entity in entities:
                query = f"MERGE (n:{entity.type} {{name: $name}})"
                session.run(query, name=entity.name)
            
            # 写入控股关系
            for rel in relationships:
                query = f"""
                MATCH (a {{name: $source}})
                MATCH (b {{name: $target}})
                MERGE (a)-[:{rel.type}]->(b)
                """
                session.run(query, source=rel.source, target=rel.target)
    def get_graph(self, parent_company: str):
        """使用图遍历算法查找所有子公司及路径"""
        with self.driver.session() as session:
            # 使用Cypher的路径查询功能实现多跳推理
            query = """
            MATCH path = (parent:Company {name: $parent_name})-[:CONTROLS*1..]->(subsidiary:Company)
            RETURN subsidiary.name as subsidiary, 
                   length(path) as depth,
                   [node in nodes(path) | node.name] as path_nodes
            ORDER BY depth, subsidiary.name
            """
            
            result = session.run(query, parent_name=parent_company)
            subsidiaries = []
            for record in result:
                subsidiaries.append({
                    'subsidiary': record['subsidiary'],
                    'depth': record['depth'],
                    'path': record['path_nodes']
                })
            
            return subsidiaries


if __name__ == "__main__":
      # 连接数据库
    driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "neo4j2025"))
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    # 初始化LLM
    llm  = OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1", 
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        is_chat_model=True
    )

    ## init_data 
     # 公司控股关系演示文本
    text = """
        A公司是一家大型集团公司。
        A公司控股B公司，持股比例为60%。
        A公司还控股D公司，持股比例为55%。
        B公司控股C公司，持股比例为70%。
        B公司控股E公司，持股比例为80%。
        C公司控股F公司，持股比例为65%。
        D公司控股G公司，持股比例为75%。
        """
    entity_extractor = ExtractEntities(llm)
    entities = entity_extractor.get_entities_by_llm(text)
    relationships = entity_extractor.extract_relationships(text, entities)

    # store entities and releationship
    db_operator = DbOperation(driver)
    db_operator.store_entities_to_db(entities, relationships)

    question = "我想知道 B公司有多少子公司"
    # question = 'B公司的被什么公司控股'
    question_entity = entity_extractor.get_entity_from_question(question)
    print(question_entity)
    subsidiaries = db_operator.get_graph(question_entity['name'])

    prompt = f"""
        你是一个实体解析专家, 善于根据以下图数据库的查询结构
        {json.dumps(subsidiaries)}
        中,回答用户的问题 {question}
        """
    # prompt = f"""
    #     你是一个实体解析专家, 善于根据以下的文本
    #     {text}
    #     中,回答用户的问题 {question}
    #     """
    response = llm.complete(prompt)
    print(response)