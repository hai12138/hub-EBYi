# -*- coding: utf-8 -*-
"""
BERT 中文文本分类（本地 bert-base-chinese 微调）
数据：cnews.train.txt（两列 tab 分隔：label \t text 或 text \t label）
目标：训练后输入新样本，输出预测类别
"""

import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


# =========================
# 0) 配置区（只改这里）
# =========================
DATA_PATH = "cnews.train.txt"  # 训练集：两列tab分隔
MODEL_DIR = "G:/python-AI/models/bert-base-chinese"  # 本地 bert-base-chinese 目录

MAX_SAMPLES = 2000        # 为了快速跑通：200 / 500 / 1000 / 2000
TEST_SIZE = 0.2
RANDOM_STATE = 42

MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 1
LR = 2e-5

OUTPUT_DIR = "./results_bert_cnews"


# =========================
# 1) 读取数据 & 自动判断 label/text 列
# =========================
def load_cnews_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, engine="python").dropna()
    if df.shape[1] < 2:
        raise ValueError("数据列数不足：cnews.train.txt 需要至少两列（label 和 text），用 tab 分隔。")
    return df


df = load_cnews_tsv(DATA_PATH)

# 自动判断哪列是 label：label 唯一值通常更少
col0_unique = df[0].nunique()
col1_unique = df[1].nunique()
if col0_unique < col1_unique:
    label_col, text_col = 0, 1
else:
    label_col, text_col = 1, 0

# 抽样加速（随机抽样，不要用 [:N] 只取开头）
if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

texts = df[text_col].astype(str).tolist()
raw_labels = df[label_col].astype(str).tolist()

# LabelEncoder：把文本标签转成 0..(num_classes-1)
lbl = LabelEncoder()
labels = lbl.fit_transform(raw_labels)
num_classes = len(lbl.classes_)

print(f"[INFO] total_samples={len(df)}  num_classes={num_classes}")
print(f"[INFO] label examples={list(lbl.classes_)[:10]}")

# 分割训练/验证（分层抽样）
x_train, x_eval, y_train, y_eval = train_test_split(
    texts,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

print(f"[INFO] train={len(x_train)} eval={len(x_eval)}")


# =========================
# 2) tokenizer + model（本地加载）
# =========================
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"MODEL_DIR 不是目录或不存在：{MODEL_DIR}")

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=num_classes)


# =========================
# 3) 文本编码 -> HuggingFace Dataset
# =========================
def encode_texts(text_list):
    return tokenizer(
        text_list,
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

train_enc = encode_texts(x_train)
eval_enc = encode_texts(x_eval)

train_ds = Dataset.from_dict({
    "input_ids": train_enc["input_ids"],
    "attention_mask": train_enc["attention_mask"],
    "labels": y_train
})

eval_ds = Dataset.from_dict({
    "input_ids": eval_enc["input_ids"],
    "attention_mask": eval_enc["attention_mask"],
    "labels": y_eval
})


# =========================
# 4) 指标函数
# =========================
def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == label_ids).mean()
    return {"accuracy": acc}


# =========================
# 5) TrainingArguments + Trainer
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    # 小样本快速验证更合理的 warmup 写法：用比例别写死 500
    warmup_ratio=0.1,
    weight_decay=0.01,

    # 日志/评估/保存策略（尽量快）
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="no",     # 先不保存checkpoint，加速
    report_to="none",

    # 如果你用 GPU，可减少 CPU 负担
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics
)

# =========================
# 6) 训练 + 验证
# =========================
trainer.train()
metrics = trainer.evaluate()
print("[EVAL]", metrics)


# =========================
# 7) 新样本预测（你要的验证分类效果）
# =========================
def predict_one(text: str) -> str:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = int(outputs.logits.argmax(dim=-1).item())
        return lbl.inverse_transform([pred_id])[0]



test_text = "中国男篮在亚洲杯比赛中取得胜利"  # 随便写一句
print("[PRED]", predict_one(test_text))
