import pandas as pd
import torch
import numpy as np
import joblib  # [新增] 用于保存标签编码器
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch.nn.functional as F

# 1. 直接读取 txt，指定分隔符为 _!_
df = pd.read_csv(
    "toutiao_cat_data.txt",
    sep="_!_",
    header=None,
    engine="python", # 必须加这个，因为 _!_ 是多字符分隔符
    names=["id", "code", "category", "text", "keywords"] # 给5列分别起名字
)
label2id = {
    "news_story": 0,         # 民生/故事
    "news_culture": 1,       # 文化
    "news_entertainment": 2, # 娱乐
    "news_sports": 3,        # 体育
    "news_finance": 4,       # 财经
    "news_house": 5,         # 房产
    "news_car": 6,           # 汽车
    "news_edu": 7,           # 教育
    "news_tech": 8,          # 科技
    "news_military": 9,      # 军事
    "news_travel": 10,       # 旅游
    "news_world": 11,        # 国际
    "stock": 12,             # 股票/证券
    "news_agriculture": 13,  # 三农/农业
    "news_game": 14          # 游戏/电竞
}
# 2. 反向映射表：用于模型配置 (预测时，模型输出 3，查表得知是 news_sports)
id2label = {v: k for k, v in label2id.items()}

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[["text", "category"]] # 只取这两列
df["label_id"] = df["category"].map(label2id)

# 转成列表传给 dataset
texts = df["text"].values.tolist()
labels = df["label_id"].values.tolist()


# lbl = LabelEncoder()
# 拟合数据 (为了演示速度，这里还是取前500个，实际使用建议取全部)
# labels = lbl.fit_transform(df["category"].values[:500])
# texts = list(df["text"].values[:500])
small_df = df.iloc[:500]
texts = small_df["text"].values.tolist()
labels = small_df["label_id"].values.tolist()
# 保存标签编码器，这步非常重要！预测时要用！
# joblib.dump(lbl, "label_encoder.pkl")
# print("标签编码器已保存至 label_encoder.pkl")

# 分割数据
x_train, x_test, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=None
)


# 2. 模型与分词器初始化

model_path = "./bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(label2id),  # 自动算出是 15 类
    label2id=label2id,         # 存入 config
    id2label=id2label,        # 存入 config
    ignore_mismatched_sizes=True
)


# 3. 数据集转换

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})


# 4. 训练设置

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # 演示用，设小一点
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy" # 指定根据准确率选最好的模型
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)


# 5. 开始训练与保存

print("开始训练...")
trainer.train()

class TextClassifier:
    def __init__(self, tokenizer, model):
        """
        初始化分类器
        :param model_dir: 训练好的模型路径
        :param label_encoder_path: 保存的 label_encoder 路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在使用设备: {self.device}")



        # 2. 加载分词器和模型
        print("正在加载模型权重...")
        self.tokenizer = tokenizer
        self.model = model

        self.model.to(self.device)
        self.model.eval()  # 切换到评估模式 (关闭 Dropout)
        print("模型加载完成！")

    def predict(self, text):
        """
        对单条文本进行预测
        """
        # 预处理
        inputs = self.tokenizer(
            text,
            return_tensors="pt",  # 返回 PyTorch 张量
            padding=True,
            truncation=True,
            max_length=64
        )

        # 移动到 GPU/CPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理 (不计算梯度)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 计算概率
        probs = F.softmax(logits, dim=-1)

        # 获取最大概率的索引
        pred_idx = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][pred_idx].item()

        # 将索引转换为原始标签文本
        pred_label = id2label[pred_idx]

        return pred_label, confidence



if __name__ == "__main__":
    # 实例化分类器
    classifier = TextClassifier(tokenizer, model)

    # 模拟一些新数据
    test_sentences = [
        "姚明成为中国篮球主席",
        "李白到底是哪个国家的人",
        "杨幂和关晓彤什么关系"
    ]

    print("-" * 30)
    for text in test_sentences:
        label, conf = classifier.predict(text)
        print(f"文本: {text}")
        print(f"预测: {label} (置信度: {conf:.2%})")
        print("-" * 30)
