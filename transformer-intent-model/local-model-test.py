import torch
import numpy as np
import os

def load_local_intent_model(model_dir="./intent_model/best_model"):
    """
    加载本地保存的意图识别模型
    :param model_dir: 本地模型保存目录
    :return: 加载后的模型、分词器、标签映射
    """
    # 1. 加载模型（自动读取 config.json 和 pytorch_model.bin）
    from transformers import BertForSequenceClassification, BertTokenizerFast

    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)

    # 2. 加载标签映射
    label_mapping = np.load(os.path.join(model_dir, "label_mapping.npy"), allow_pickle=True).item()
    id2label = label_mapping["id2label"]
    classes = label_mapping["classes"]

    max_length = 128
    # 3. 加载训练配置（可选）
    # import json
    # with open(os.path.join(model_dir, "training_config.json"), "r", encoding="utf-8") as f:
    #     training_config = json.load(f)
    # max_length = training_config["max_length"]

    # 4. 设置设备（与训练时一致）
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 切换到评估模式

    print(f"模型加载完成！")
    print(f"设备: {device}")
    print(f"支持的意图类别: {classes}")
    print(f"最大文本长度: {128}")

    return model, tokenizer, id2label, max_length, device


def local_model_predict(text, model, tokenizer, id2label, max_length, device):
    """使用加载的本地模型进行意图预测"""
    # 文本预处理
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # 推理（禁用梯度计算，节省资源）
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=1).item()
        intent = id2label[predicted_id]
        confidence = torch.softmax(logits, dim=1)[0][predicted_id].item()

    return {
        "text": text,
        "intent": intent,
        "confidence": round(confidence, 4)
    }


# ---------------------- 加载模型并测试 ----------------------
if __name__ == "__main__":
    # 1. 加载本地模型
    model, tokenizer, id2label, max_length, device = load_local_intent_model(
        model_dir="/Users/yegaosong/develop/intent_model/best_model"  # 本地模型目录
    )

    # 2. 测试预测
    test_texts = [
        "杭州后天会下雪吗？",
        "帮我订一张周末去深圳的高铁票",
        "有没有最新的科技新闻？",
        "播放一首陈奕迅的歌"
    ]

    print("\n=== 本地模型预测结果 ===")
    for text in test_texts:
        result = local_model_predict(text, model, tokenizer, id2label, max_length, device)
        print(f"文本: {result['text']}")
        print(f"预测意图: {result['intent']} (置信度: {result['confidence']})")
        print("-" * 50)