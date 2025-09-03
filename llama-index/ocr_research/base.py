from logging import raiseExceptions
from math import e
import select
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import os
from typing import List, Union

class ImageOCRReader(BaseReader):
    """使用 PP-OCR v5 从图像中提取文本并返回 Document"""
    
    def __init__(self, lang='ch', use_gpu=False, **kwargs):
        """
        Args:
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu: 是否使用 GPU 加速
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "`paddleocr` package not found, please run `pip install paddleocr`"
            )

        self.lang = lang
        self.use_gpu = use_gpu
        self.reader_kwargs = kwargs


    def parseOcrResponse(self, result, file):
        documents = []
        num = 1
        for i, res in enumerate(result):
            res_dict = res._to_json()
            if not res_dict or not res_dict["res"]:
                continue
            if not res_dict["res"]["rec_texts"]:
                continue
            merged_text = '\n'.join(res_dict["res"]["rec_texts"])
            if res_dict["res"]["rec_scores"]: 
                rec_scores = res_dict["res"]["rec_scores"]
                average_score = sum(rec_scores) / len(rec_scores)
            # 创建Document对象（包含元数据）
            document = Document(
                text=merged_text,
                metadata={
                    "image_path": res_dict["res"]["input_path"],  # 图片路径
                    "image_path": file,  # 图片完整路径
                    "ocr_model": "PP-OCRv5",  # 识别引擎信息
                    "language": self.lang,
                    "num_text_blocks": num,
                    "avg_confidence": average_score
                }
            )
            num += 1
            documents.append(document)
            
        return documents
            
    def load_data(self, file: Union[str, List[str]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表
        Args:
            file: 图像路径字符串 或 路径列表
        Returns:
            List[Document]
        """
        # 实现 OCR 提取逻辑
        # 将每张图的识别结果拼接成文本
        # 构造 Document 对象，附带元数据（如 image_path, ocr_confidence_avg）

         # 1. 验证参数类型是否符合 Union[str, List[str]]
        if not isinstance(file, (str, list)):
            raise TypeError(
                f"参数类型错误：期望 str 或 List[str]，实际得到 {type(file).__name__}"
            )
        
        # 2. 如果是列表，验证列表元素是否全为字符串
        if isinstance(file, list):
            for idx, item in enumerate(file):
                if not isinstance(item, str):
                    raise TypeError(
                        f"列表元素类型错误：第 {idx+1} 个元素应为 str，实际为 {type(item).__name__}"
                    )
        
        # 3. （可选）验证文件路径是否有效（如是否存在）
        files_to_load = [file] if isinstance(file, str) else file
        for f in files_to_load:
            if not os.path.exists(f):
                raise FileNotFoundError(f"文件不存在：{f}")
        try:
            from paddleocr import PaddleOCR
        
            docs = []
            device = "gpu" if self.use_gpu else None
            ocr = PaddleOCR(
                use_doc_orientation_classify=False, 
                use_doc_unwarping=False, 
                use_textline_orientation=False,
                lang=self.lang,  
                device=device
            ) 
            for i, file in enumerate(files_to_load):
                result = ocr.predict(file)
                single_file_docs = self.parseOcrResponse(result, file)
                if single_file_docs:
                    docs.extend(single_file_docs)
            return docs
        except Exception:
            raiseExceptions(
                "PaddleOCR predict image error."
            )