from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import StringPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.llms import Tongyi
import dotenv
dotenv.load_dotenv()

# 项目报告数据模型
class ProjectReport(BaseModel):
    project_name: str = Field(description="项目名称")
    progress: str = Field(description="进度状态")
    completed_tasks: List[str] = Field(description="已完成任务")
    pending_tasks: List[str] = Field(description="待完成任务")
    risks: List[str] = Field(description="风险点")

# 自定义项目报告模板
class ProjectReportTemplate(StringPromptTemplate):
    """项目报告自定义模板"""
    
    language_style: str = Field(description="语言风格", default="professional")
    
    def __init__(self, **kwargs):
        kwargs.setdefault('input_variables', ['input_text'])
        super().__init__(**kwargs)
    
    def format(self, **kwargs: Any) -> str:
        """格式化项目报告模板"""
        input_text = kwargs.get("input_text", "")
        
        return f"""作为项目经理，请分析以下项目信息并生成状态报告：

                项目信息：
                {input_text}

                请按以下JSON格式输出项目报告：
                {{
                "project_name": "项目名称",
                "progress": "进度状态",
                "completed_tasks": ["已完成任务1", "已完成任务2"],
                "pending_tasks": ["待完成任务1", "待完成任务2"],
                "risks": ["风险点1", "风险点2"]
                }}

                报告要求：
                1. 客观评估项目当前状态
                2. 识别关键里程碑和交付物
                3. 预警潜在风险和阻碍
                4. 采用{self.language_style}的汇报风格"""

# 输出解析器
class ProjectParser(BaseOutputParser[Dict[str, Any]]):
    """项目报告解析器"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        import json
        import re
        
        # 提取JSON

        print("text", text)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        json_str = json_match.group(0) if json_match else text
        
        # 解析并验证
        data = json.loads(json_str)
        validated = ProjectReport(**data)
        print("validated", validated)
        return validated.model_dump()
    
    def get_format_instructions(self) -> str:
        return "请以JSON格式输出项目报告"

# 调用通义千问
def demo_with_tongyi():
    """使用通义千问实际调用演示"""
    print("项目报告自定义模板 - 通义千问实际调用")
    print("-" * 40)
    
    # 初始化通义千问
    llm = Tongyi(
        model_name="qwen-max",
        temperature=0.1,
        top_p=0.8
    )
    
    # 创建模板和解析器
    template = ProjectReportTemplate(language_style="executive")
    parser = ProjectParser()
    
    # 测试项目信息
    project_info = """
    AI客服系统开发项目，启动3个月
    已完成：需求分析、技术选型、基础架构搭建、自然语言处理模块
    正在进行：对话管理系统、知识库集成
    计划完成：用户界面开发、系统测试、部署上线
    当前问题：知识库数据质量不够，影响回答准确性
    """
    
    # 生成提示
    prompt = template.format(input_text=project_info)
    print("生成的提示:")
    print(prompt[:200] + "...")
    
    # 实际调用通义千问
    print("\n正在调用通义千问...")
    response = llm.invoke(prompt)
    print("\n通义千问响应:")
    print(response)
    
    # 解析结果
    print("\n解析结果:")
    result = parser.parse(response)
    for key, value in result.items():
        print(f"  {key}: {value}")

# 创建完整的处理链
def create_project_chain():
    """创建项目报告处理链"""
    llm = Tongyi(model_name="qwen-max", temperature=0.1)
    template = ProjectReportTemplate(language_style="professional")
    parser = ProjectParser()
    
    # 创建链式处理
    chain = template | llm | parser
    return chain

def demo_chain():
    """演示链式处理"""
    print("\n链式处理演示")
    print("-" * 30)
    
    chain = create_project_chain()
    
    project_info = """
    移动应用开发项目，进度60%
    完成：UI设计、用户注册登录、核心功能开发
    待完成：支付功能、推送通知、性能优化
    风险：iOS审核可能延期，安卓兼容性问题
    """
    
    print("输入项目信息:")
    print(project_info)
    
    print("\n执行链式处理...")
    result = chain.invoke({"input_text": project_info})
    
    print("最终结果:")
    for key, value in result.items():
        print(f"  {key}: {value}")

def main():
    """主函数"""
    # demo_with_tongyi()
    demo_chain()

if __name__ == "__main__":
    main()