"""
企业级 LangChain 应用 - 智能客服系统
兼容当前版本，包含完整的错误处理和容错机制
"""
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from typing import Dict, List, Optional, Any
import json
import logging
import time
from functools import wraps

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_with_backoff(max_attempts=3, base_delay=1.0):
    """自定义重试装饰器，实现指数退避"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"重试失败，已达到最大尝试次数: {e}")
                        raise e
                    
                    delay = base_delay * (2 ** attempt)  # 指数退避
                    logger.warning(f"第{attempt + 1}次尝试失败，{delay}秒后重试: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def timeout_handler(timeout_seconds=30.0):
    """超时控制装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_signal_handler(signum, frame):
                raise TimeoutError(f"操作超时 ({timeout_seconds}秒)")
            
            # 设置超时信号
            old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # 取消超时
                return result
            except TimeoutError:
                logger.error(f"操作超时: {timeout_seconds}秒")
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

class CustomerServiceResponse(BaseOutputParser[Dict]):
    """客服响应解析器"""
    
    def parse(self, text: str) -> Dict:
        try:
            # 尝试解析JSON格式
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                json_str = text[start:end]
                return json.loads(json_str)
            
            # 如果不是JSON，返回简单格式
            return {
                "response": text.strip(),
                "category": "general",
                "confidence": 0.8,
                "requires_human": False
            }
        except Exception as e:
            raise OutputParserException(f"解析失败: {e}")
    
    def get_format_instructions(self) -> str:
        return """请以JSON格式回复：
            {
                "response": "回复内容",
                "category": "问题类别(technical/billing/general)",
                "confidence": 0.9,
                "requires_human": false
            }"""

class EnterpriseCustomerService:
    """企业级客服系统"""
    
    def __init__(self):
        self.setup_models()
        self.setup_chains()
        self.setup_fallback_system()
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
    
    def setup_models(self):
        """设置模型"""
        # 主要模型 - 高性能
        self.primary_model = Tongyi(
            model_name="qwen-max",
            temperature=0.3,
            max_tokens=500
        )
        
        # 备用模型 - 稳定性优先
        self.backup_model = Tongyi(
            model_name="qwen-plus", 
            temperature=0.1,
            max_tokens=300
        )
    
    def setup_chains(self):
        """设置处理链"""
        self.parser = CustomerServiceResponse()
        
        # 技术问题处理链
        tech_prompt = PromptTemplate(
            input_variables=["question", "user_info"],
            template="""你是技术支持专家，请回答用户的技术问题。

                    用户信息：{user_info}
                    问题：{question}

                    {format_instructions}

                    请提供专业的技术解答。""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # 账单问题处理链
        billing_prompt = PromptTemplate(
            input_variables=["question", "user_info"],
            template="""你是账单客服专员，请处理用户的账单相关问题。

                用户信息：{user_info}
                问题：{question}

                {format_instructions}

                请提供准确的账单信息和解决方案。""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # 通用问题处理链
        general_prompt = PromptTemplate(
            input_variables=["question", "user_info"],
            template="""你是客服代表，请友好地回答用户问题。

                用户信息：{user_info}
                问题：{question}

                {format_instructions}

                请提供有帮助的回复。""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # 创建处理链
        self.tech_chain = tech_prompt | self.primary_model | self.parser
        self.billing_chain = billing_prompt | self.primary_model | self.parser
        self.general_chain = general_prompt | self.primary_model | self.parser
    
    def setup_fallback_system(self):
        """设置容错系统"""
                # 创建智能路由分支
        self.smart_router = RunnableBranch(
            (self._is_technical_question, self.tech_chain_with_fallback),
            (self._is_billing_question, self.billing_chain_with_fallback),
            self.general_chain_with_fallback  # 默认分支
        )
        # 创建带有回退机制的处理函数
        def create_fallback_chain(primary_chain, chain_name):
            def fallback_processor(input_data):
                try:
                    # 第一层：主要链处理
                    return primary_chain.invoke(input_data)
                except Exception as e:
                    logger.warning(f"{chain_name} 主链失败，尝试备用模型: {e}")
                    try:
                        # 第二层：备用模型处理
                        backup_chain = primary_chain.first | self.backup_model | self.parser
                        return backup_chain.invoke(input_data)
                    except Exception as e2:
                        logger.error(f"{chain_name} 备用模型失败，使用简单响应: {e2}")
                        # 第三层：简单响应
                        return {
                            "response": "抱歉，系统暂时繁忙，请稍后重试或联系人工客服。",
                            "category": "system_error",
                            "confidence": 1.0,
                            "requires_human": True
                        }
            
            return RunnableLambda(fallback_processor)
        
        # 为每个链添加回退机制
        self.tech_chain_with_fallback = create_fallback_chain(self.tech_chain, "技术支持")
        self.billing_chain_with_fallback = create_fallback_chain(self.billing_chain, "账单服务")
        self.general_chain_with_fallback = create_fallback_chain(self.general_chain, "通用服务")
        

    
    def _is_technical_question(self, x: Dict) -> bool:
        """判断是否为技术问题"""
        question = x.get("question", "").lower()
        tech_keywords = ["bug", "错误", "故障", "技术", "API", "代码", "系统", "登录", "密码"]
        return any(keyword in question for keyword in tech_keywords)
    
    def _is_billing_question(self, x: Dict) -> bool:
        """判断是否为账单问题"""
        question = x.get("question", "").lower()
        billing_keywords = ["账单", "费用", "付款", "充值", "退款", "价格", "订单"]
        return any(keyword in question for keyword in billing_keywords)
    
    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    @timeout_handler(timeout_seconds=30.0)
    def _process_with_retry_and_timeout(self, input_data: Dict) -> Dict:
        """带重试和超时的处理方法"""
        return self.smart_router.invoke(input_data)
    
    def process_customer_inquiry(self, question: str, user_info: Dict) -> Dict:
        """处理客户咨询"""
        start_time = time.time()
        self.performance_stats["total_requests"] += 1
        
        try:
            logger.info(f"处理客户咨询: {question[:50]}...")
            
            # 准备输入
            input_data = {
                "question": question,
                "user_info": json.dumps(user_info, ensure_ascii=False)
            }
            
            # 执行带重试和超时的处理
            result = self._process_with_retry_and_timeout(input_data)
            
            # 添加处理时间和状态
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time
            result["status"] = "success"
            
            # 更新性能统计
            self.performance_stats["successful_requests"] += 1
            self._update_average_response_time(processing_time)
            
            logger.info(f"处理完成，耗时: {processing_time}秒")
            return result
            
        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            self.performance_stats["failed_requests"] += 1
            
            logger.error(f"处理失败: {e}")
            return {
                "response": "系统出现异常，请联系技术支持。",
                "category": "system_error",
                "confidence": 0.0,
                "requires_human": True,
                "status": "error",
                "error": str(e),
                "processing_time": processing_time
            }
    
    def _update_average_response_time(self, new_time: float):
        """更新平均响应时间"""
        total_successful = self.performance_stats["successful_requests"]
        current_avg = self.performance_stats["average_response_time"]
        
        # 计算新的平均值
        new_avg = ((current_avg * (total_successful - 1)) + new_time) / total_successful
        self.performance_stats["average_response_time"] = round(new_avg, 2)
    
    def batch_process_inquiries(self, inquiries: List[Dict]) -> List[Dict]:
        """批量处理客户咨询"""
        logger.info(f"开始批量处理 {len(inquiries)} 个咨询")
        
        results = []
        for inquiry in inquiries:
            result = self.process_customer_inquiry(
                inquiry["question"],
                inquiry.get("user_info", {})
            )
            results.append(result)
        
        logger.info(f"批量处理完成")
        return results
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        if stats["total_requests"] > 0:
            stats["success_rate"] = round(
                (stats["successful_requests"] / stats["total_requests"]) * 100, 2
            )
        else:
            stats["success_rate"] = 0.0
        
        return stats

def demo_enterprise_application():
    """企业应用演示"""
    print("=== 企业级 LangChain 客服系统演示 ===")
    
    # 初始化系统
    customer_service = EnterpriseCustomerService()
    
    # 测试用例
    test_cases = [
        {
            "question": "我的API调用出现500错误，怎么解决？",
            "user_info": {"user_id": "12345", "plan": "企业版", "region": "北京"}
        },
        {
            "question": "我想查看本月的账单详情",
            "user_info": {"user_id": "67890", "plan": "标准版", "region": "上海"}
        },
        {
            "question": "你们的服务怎么样？",
            "user_info": {"user_id": "11111", "plan": "免费版", "region": "深圳"}
        }
    ]
    
    # 单个处理演示
    print("\n--- 单个处理演示 ---")
    for i, case in enumerate(test_cases, 1):
        print(f"\n客户咨询 {i}:")
        print(f"问题: {case['question']}")
        print(f"用户信息: {case['user_info']}")
        
        result = customer_service.process_customer_inquiry(
            case["question"], 
            case["user_info"]
        )
        
        print(f"回复: {result['response']}")
        print(f"类别: {result['category']}")
        print(f"置信度: {result['confidence']}")
        print(f"需要人工: {result['requires_human']}")
        print(f"处理时间: {result['processing_time']}秒")
        print(f"状态: {result['status']}")
    
    # 批量处理演示
    print(f"\n--- 批量处理演示 ---")
    batch_results = customer_service.batch_process_inquiries(test_cases)
    
    print(f"批量处理了 {len(batch_results)} 个咨询")
    for i, result in enumerate(batch_results, 1):
        print(f"结果 {i}: {result['category']} - {result['response'][:50]}...")
    
    # 性能统计
    print(f"\n--- 性能统计 ---")
    stats = customer_service.get_performance_stats()
    print(f"总请求数: {stats['total_requests']}")
    print(f"成功请求数: {stats['successful_requests']}")
    print(f"失败请求数: {stats['failed_requests']}")
    print(f"成功率: {stats['success_rate']}%")
    print(f"平均响应时间: {stats['average_response_time']}秒")

if __name__ == "__main__":
    demo_enterprise_application()
