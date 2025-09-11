import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import Dataset, DatasetDict
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(os.path.dirname(current_dir), 'data') 
# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 统一处理所有设备的随机种子
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

class BERTIntentClassifier:
    def __init__(self, model_name="bert-base-uncased", max_length=128):
        """
        初始化意图分类器
        :param model_name: HuggingFace上的BERT模型名称
        :param max_length: 文本最大长度
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_labels = 0
        
        # 创建保存模型和数据的目录
        self.model_dir = f"/Users/yegaosong/develop/intent_model"
        self.data_dir = f"/Users/yegaosong/develop/data"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 统一设备管理：优先MPS，其次CUDA，最后CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"使用计算设备: {self.device}")


    def load_data(self, data_path=None, sample_data=False):
        """
        加载数据，如果没有提供数据路径，使用示例数据
        :param data_path: 数据文件路径
        :param sample_data: 是否使用示例数据
        """
        if sample_data or data_path is None:
            # 生成示例数据
            data = {
                "text": [
                    "明天北京天气怎么样", "上海后天的气温是多少", "广州会下雨吗",
                    "帮我订一张去上海的机票", "预订明天到北京的酒店", "我要租车",
                    "今天有什么新闻", "体育新闻汇总", "科技方面的最新消息",
                    "播放周杰伦的歌", "我想听轻音乐", "暂停播放"
                ],
                "label": [
                    "weather", "weather", "weather",
                    "booking", "booking", "booking",
                    "news", "news", "news",
                    "music", "music", "music"
                ]
            }
            self.df = pd.DataFrame(data)
            # 保存示例数据供后续使用
            self.df.to_csv(os.path.join(self.data_dir, "sample_intent_data.csv"), index=False)
        else:
            # 从CSV文件加载数据
            self.df = pd.read_csv(data_path)
            
        print(f"加载数据完成，共 {len(self.df)} 条样本")
        print(f"意图类别分布：\n{self.df['label'].value_counts()}")

    def preprocess_data(self):
        """数据预处理：标签编码、划分训练集和验证集、转换为Dataset格式"""
        # 标签编码：将文本标签转换为数字
        self.df["label"] = self.label_encoder.fit_transform(self.df["label"])
        self.num_labels = len(self.label_encoder.classes_)
        
        # 保存标签映射，用于后续推理
        label_mapping = {
            "id2label": {i: label for i, label in enumerate(self.label_encoder.classes_)},
            "label2id": {label: i for i, label in enumerate(self.label_encoder.classes_)}
        }
        np.save(os.path.join(self.model_dir, "label_mapping.npy"), label_mapping)
        
        # 划分训练集和验证集（使用比例而非固定数量，更灵活）
        train_df, val_df = train_test_split(
            self.df, 
            test_size=4,  # 使用20%作为验证集
            random_state=42,
            stratify=self.df["label"]  # 保持标签分布一致
        )
        
        # 转换为Hugging Face Dataset格式
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        self.dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        # 应用分词器进行文本预处理
        self.tokenized_dataset = self.dataset.map(
            self._tokenize_function, 
            batched=True,
            remove_columns=["text", "__index_level_0__"]  # 移除不需要的列
        )
        
        # 重命名标签列，符合模型要求
        self.tokenized_dataset = self.tokenized_dataset.rename_column("label", "labels")
        
        # 设置为PyTorch张量格式
        self.tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        print("数据预处理完成")

    def _tokenize_function(self, examples):
        """内部使用的分词函数"""
        return self.tokenizer(
            examples["text"],
            truncation=True,  # 截断过长文本
            padding="max_length",  # 填充至最大长度
            max_length=self.max_length  # 文本最大长度
        )

    def load_model(self):
        """加载预训练模型并添加分类头"""
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label={i: label for i, label in enumerate(self.label_encoder.classes_)},
            label2id={label: i for i, label in enumerate(self.label_encoder.classes_)}
        )
        
        # 将模型移至统一设备
        self.model = self.model.to(self.device)
        print(f"模型已加载到 {self.device}")

    def train(self):
        """训练模型"""
        # 定义训练参数
        training_args = TrainingArguments(
            output_dir=self.model_dir,  # 模型保存路径
            learning_rate=2e-5,  # BERT模型常用学习率
            per_device_train_batch_size=8,  # 每个设备的训练批次大小
            per_device_eval_batch_size=8,   # 每个设备的评估批次大小
            num_train_epochs=5,  # 训练轮数
            weight_decay=0.01,  # 权重衰减，防止过拟合
            save_strategy="epoch",  # 每轮保存一次模型
            eval_strategy="epoch",  # 与保存策略保持一致
            load_best_model_at_end=True,  # 训练结束后加载最佳模型
            logging_dir="./logs",  # 日志保存路径
            logging_steps=10,  # 每10步记录一次日志
            report_to="none",  # 不使用wandb等报告工具
            # 确保训练时使用正确的设备
            fp16=False  # MPS不支持fp16，禁用混合精度训练
        )
        
        # 数据整理器
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 定义评估指标计算函数
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            return {
                "accuracy": (predictions == labels).mean()
            }
        
        # 初始化Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # 开始训练
        print("开始训练...")
        trainer.train()
        
        # 1. 保存最佳模型（由 load_best_model_at_end=True 保证）
        best_model_dir = os.path.join(self.model_dir, "best_model")
        trainer.save_model(best_model_dir)  # 保存模型权重、配置文件
        print(f"最佳模型已保存至: {best_model_dir}")

        # 2. 单独保存分词器（确保后续推理时使用相同的分词逻辑）
        self.tokenizer.save_pretrained(best_model_dir)
        print(f"分词器已保存至: {best_model_dir}")

        # 3. 保存标签映射（用于将数字标签转回文本意图）
        label_mapping = {
            "id2label": self.model.config.id2label,
            "label2id": self.model.config.label2id,
            "classes": self.label_encoder.classes_  # 原始标签列表
        }
        np.save(os.path.join(best_model_dir, "label_mapping.npy"), label_mapping)
        print(f"标签映射已保存至: {best_model_dir}/label_mapping.npy")
     

    def evaluate(self):
        """评估模型性能"""
        # 获取验证集数据加载器
        val_dataloader = torch.utils.data.DataLoader(
            self.tokenized_dataset["validation"],
            batch_size=8,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
        
        # 模型切换到评估模式
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        # 不计算梯度，节省内存
        with torch.no_grad():
            for batch in val_dataloader:
                # 将数据移至统一设备（关键修复）
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 模型推理
                outputs = self.model(** batch)
                logits = outputs.logits
                
                # 获取预测结果
                predictions = torch.argmax(logits, dim=1)
                
                # 收集结果（移回CPU用于后续处理）
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        # 打印详细评估报告
        print("\n评估结果：")
        print(classification_report(
            all_labels,
            all_predictions,
            target_names=self.label_encoder.classes_
        ))

    def predict(self, text):
        """使用训练好的模型进行意图预测"""
        # 文本预处理
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 将数据移至统一设备（关键修复）
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 模型推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 解析结果
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        intent = self.model.config.id2label[predicted_class_id]
        confidence = torch.softmax(logits, dim=1)[0][predicted_class_id].item()
        
        return {
            "text": text,
            "intent": intent,
            "confidence": round(confidence, 4)
        }

if __name__ == "__main__":
    # 初始化分类器，使用小型BERT模型减少参数量
    classifier = BERTIntentClassifier(
        model_name="prajjwal1/bert-small",  # 小型BERT模型，66M参数
        max_length=64  # 意图识别文本通常较短，减小最大长度
    )
    
    # 加载数据（使用示例数据）
    classifier.load_data(sample_data=True)
    
    # 预处理数据
    classifier.preprocess_data()
    
    # 加载模型
    classifier.load_model()
    
    # 训练模型
    classifier.train()
    
    # 评估模型
    classifier.evaluate()
    
    # 测试预测
    test_texts = [
        "北京明天会下雨吗",
        "帮我订一张机票",
        "今天有什么热点新闻",
        "播放一首流行歌曲"
    ]
    
    print("\n预测示例：")
    for text in test_texts:
        result = classifier.predict(text)
        print(f"文本: {result['text']}")
        print(f"预测意图: {result['intent']} (置信度: {result['confidence']})")
        print("---")
