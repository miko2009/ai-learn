# 文本分块技术

## 1. 文本分块的原理和重要性
- **目的**：将长文本分解成适当大小的片段，便于嵌入、索引和存储，提高检索精确度
- **重要性**：避免主题稀释，确保检索过程能精准捕捉每个主题的核心信息

## 2. 分块策略和方法
### 基本方法
- **CharacterTextSplitter**：按固定字符数分块
- **RecursiveCharacterTextSplitter**：递归分块，传入分割符列表按优先级使用
- **语义分块**：基于文档语义结构识别，而非简单根据空行或换行符拆分

### 智能分块策略（Unstructured）
- **Basic策略**：按最大字符数或软限制合并文本元素
- **By Title策略**：检测到新标题时关闭当前分块并开启新分块
- **By Page策略**：确保每页内容独立分块
- **By Similarity策略**：利用嵌入模型将主题相似元素组合成块

## 3. 分块实现工具
- **LangChain**：提供CharacterTextSplitter和RecursiveCharacterTextSplitter
- **LlamaIndex**：提供SemanticSplitterNodeParser进行语义分块
- **Unstructured工具**：基于文档结构分块，支持多种策略

## 4. 分块参数配置
- **chunk_size**：分块大小（如1000字符）
- **chunk_overlap**：分块重叠大小（如10字符）
- **separators**：分割符列表（如["\n\n", ".", "，", " "]）

## 5. 分块效果优化
- **避免主题稀释**：将综合信息块独立为主题明确的块
- **检索精度提升**：通过合理分块使检索过程更精准定位相关信息
- **生成质量影响**：合适的分块大小对生成质量有重要影响

## 6. 可视化与分析
- **ChunkViz工具**：用于可视化分块效果
- **断点百分位阈值**：设置分块断点（如breakpoint_percentile_threshold=95）


## 高级索引技巧
-  带滑动窗口的句子切分（Sliding Windows）
- 分块时混合生成父子文本块（Parent-Child Docs）子文本存储在爱向量数据库，父文本可存储在文件等.
- 分块时为文本块创建元数据
- 在分块时形成有级别的索引（Summary→Details ）, 提前生成假设性问题。
- 文档→嵌入对象（Document→Embedded Objects）

## 7. 应用场景示例
- **旅游知识库**：将景点信息独立分块（如"五台山：佛教名山"）
- **医疗知识库**：切分为主题明确的小块（如"高血压：常用药物"）