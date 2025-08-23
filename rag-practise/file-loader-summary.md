# 大模型文件加载工具优劣势对比

| 文件格式 | 工具 | 优势 | 劣势 | 场景 |
|----------|------|------|------|-----| 
| **文本（TXT等）** | langchain_community.document_loaders | 支持基础文本加载，与LangChain生态无缝集成，可直接对接后续处理（如分割、嵌入） | 功能简单，对复杂编码或特殊格式文本处理能力有限 |  简单文本 |
|  | llama_index | 加载后可直接用于索引构建，支持文本元数据提取，与LlamaIndex的查询功能深度适配, 比langchain 多了元数据, 如文本长度等. | 单独使用时灵活性较低，依赖自身生态 |   |
|  | unstructured | 支持多种编码格式，可自动检测文本结构（如标题、段落），输出结构化数据, 默认对文件经常分块处理 | 需额外配置，对极简文本处理效率较低 |  长且关系复杂的文本 |
| **图片（PNG/JPG等）** | langchain_community.document_loaders | 可通过 `UnstructuredImageLoader` 结合OCR（如Tesseract）提取文本，集成LangChain的OCR工具链 | 依赖外部OCR库，处理复杂图片（如多语言、模糊）效果一般 |
|  | llama_index | 支持通过插件集成OCR工具，加载后可直接生成图像描述（需模型支持），与索引流程联动 | 原生OCR能力较弱，需额外配置模型或工具 |
|  | unstructured | 内置OCR处理逻辑，支持复杂图片布局分析，对表格、多区域文本提取更精准 | 依赖Tesseract等OCR工具，安装配置较复杂 |
| **Markdown** | langchain_community.document_loaders | 有专门的 `UnstructuredMarkdownLoader`，无保留标题、列表等结构，可直接拆分节点, 变成长文本 | 对嵌套结构（如表格、代码块）处理不够细致 | 简单 md 数据，不需要结构化内容 |
|  | llama_index | 支持Markdown结构解析，可将不同层级内容映射为文档节点，优化索引效率, 保留原始内容标题,列表等结构化数据 | 对非标准Markdown语法兼容性一般 | 需要再做额外分块处理 ｜
|  | unstructured | 精准解析Markdown语法，保留所有结构（包括表格、代码块、链接），输出结构化数据, 保留原始内容标题,列表等结构化数据  | 解析速度较慢，对超大Markdown文件支持有限 | 速度慢, 功能齐整｜
| **PDF** | langchain_community.document_loaders | 提供多种加载器（如 `PyPDFLoader` 用于文本PDF，`UnstructuredPDFLoader` 用于扫描件），支持分页提取 | 扫描件PDF需额外OCR配置，复杂布局（如多列）提取易错乱 |
|  | llama_index | 支持PDF分页加载和元数据提取，可结合OCR处理扫描件，与索引构建流程深度整合 | 对复杂排版PDF的结构识别能力较弱 |
|  | unstructured | 强项：支持复杂布局PDF（多列、图表混排）、扫描件OCR，可提取表格数据，保留原始格式信息 | 处理速度较慢，依赖较多系统库（如poppler） |
| **JSON** | langchain_community.document_loaders | 有 `JSONLoader` 可指定提取字段，支持嵌套JSON解析，输出结构化文档 | 对非标准JSON（如注释、格式错误）容错率低 |
|  | llama_index | 支持JSON到文档对象的自动转换，可将JSON字段映射为文档元数据，优化检索相关性 | 复杂嵌套JSON需手动配置解析规则 |
|  | unstructured | 可直接解析JSON为结构化数据，支持JSON Lines等变体格式，输出键值对结构 | 缺乏与大模型流程的直接集成，需手动转换为文档格式 |
| **Web URL** | langchain_community.document_loaders | 提供 `WebBaseLoader` 等工具，支持静态网页提取，可结合 `BeautifulSoup` 解析，与LangChain爬虫工具链兼容 | 对动态JS渲染页面支持差，需额外集成Selenium等工具 |
|  | llama_index | 支持URL加载并自动提取核心内容（过滤广告、导航栏），可直接生成网页摘要，优化索引质量 | 对需要登录的网页处理能力有限 |
|  | unstructured | 支持复杂网页结构内容解析，可提取文本、表格、图片链接，过滤冗余内容，输出结构化数据 | 不支持动态网页渲染，需配合爬虫工具使用 | 如果动态网页, 需要用http 请求到网页内容才能进行解析｜

## 总结
- **langchain_community.document_loaders**：胜在生态集成，适合快速搭建端到端流程，对简单格式处理高效。
- **llama_index**：强于与索引/检索流程的联动，适合以"检索增强"为核心的场景。
- **unstructured**：专长于复杂格式解析和结构化提取，适合对文档结构精度要求高的场景，但配置和性能成本较高。