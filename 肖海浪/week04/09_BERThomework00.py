import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertForSequenceClassification


# ============================================================
# 0) 配置区（只改这里）
# ============================================================
DATA_PATH = "cnews.train.txt"                 # 你的数据文件：两列，tab分隔
MODEL_DIR = "G:/python-AI/models/bert-base-chinese"  # 本地 bert 目录

SAMPLE_N = 1000    # ✅ 随机抽样 N 条（不要用 [:500] 切头）
TEST_SIZE = 0.2
RANDOM_STATE = 42

MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5


# ============================================================
# 1) 读取数据
# ============================================================
df = pd.read_csv(DATA_PATH, sep="\t", header=None, engine="python").dropna()
print("raw shape:", df.shape)
print(df.head())

# ------------------------------------------------------------
# 重要：确认哪列是 text，哪列是 label
# 你的截图显示 label 像 “体育/娱乐/财经...”
# 下面这段做一个更稳的自动判断：短且重复多的是 label，长的是 text
# ------------------------------------------------------------
col0 = df[0].astype(str)
col1 = df[1].astype(str)

avg_len0, avg_len1 = col0.str.len().mean(), col1.str.len().mean()
uniq0, uniq1 = col0.nunique(), col1.nunique()

print(f"\ncol0 avg_len={avg_len0:.1f}, uniq={uniq0}")
print(f"col1 avg_len={avg_len1:.1f}, uniq={uniq1}")

# label 通常更短，唯一值更少
if (avg_len0 < avg_len1 and uniq0 < uniq1) or avg_len0 < 10:
    label_col, text_col = 0, 1
else:
    label_col, text_col = 1, 0

print(f"\n识别列：text_col={text_col}, label_col={label_col}")


# ============================================================
# 2) 随机抽样（替代 [:500]）
# ============================================================
# 用 sample 随机抽，避免“前500条全是同一类”的问题
# stratify 用 label 来保证抽样也尽量按比例
if SAMPLE_N is not None and SAMPLE_N < len(df):
    df_small = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE)
else:
    df_small = df.copy()

X = df_small[text_col].astype(str).tolist()
y_raw = df_small[label_col].astype(str).values

print("\n抽样后类别分布：")
print(pd.Series(y_raw).value_counts().head(20))


# ============================================================
# 3) 标签编码 + 分层切分（stratify 必须用 y）
# ============================================================
lbl = LabelEncoder()
y = lbl.fit_transform(y_raw)
num_labels = len(lbl.classes_)

print("\nnum_labels =", num_labels)
print("label mapping example:", {i: c for i, c in enumerate(lbl.classes_)})

# ✅ 关键：stratify 用 y（数字标签），避免字符串/编码不一致
x_train, x_test, train_label, test_label = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\ntrain size:", len(x_train), "test size:", len(x_test))


# ============================================================
# 4) Tokenizer 编码
# ============================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

train_encoding = tokenizer(
    x_train,
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN
)

test_encoding = tokenizer(
    x_test,
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN
)


# ============================================================
# 5) Dataset / DataLoader
# ============================================================
class NewsDataset(Dataset):
    """把 tokenizer 的输出 + 标签封装成 PyTorch Dataset"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_encoding, train_label)
test_dataset = NewsDataset(test_encoding, test_label)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 6) 模型 / 优化器 / 设备
# ============================================================
model = BertForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=num_labels   # ✅ 自动匹配类别数
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=LR)


# ============================================================
# 7) 准确率计算
# ============================================================
def flat_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    pred = np.argmax(logits, axis=1)
    return (pred == labels).mean()


# ============================================================
# 8) 训练 / 验证
# ============================================================
def train_one_epoch(epoch_idx: int):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        optim.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 200 == 0:
            print(f"[Epoch {epoch_idx}] step {step}/{len(train_loader)} loss={loss.item():.4f}")

    print(f"[Epoch {epoch_idx}] avg train loss={total_loss / len(train_loader):.4f}")


def evaluate():
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            logits_np = logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            total_acc += flat_accuracy(logits_np, labels_np)

    print(f"[Eval] acc={total_acc / len(test_loader):.4f}  loss={total_loss / len(test_loader):.4f}")
    print("-" * 50)


# ============================================================
# 9) 主循环
# ============================================================
for epoch in range(1, EPOCHS + 1):
    print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")
    train_one_epoch(epoch)
    evaluate()
# ============================================================
# 10) 单条文本预测函数 + 验证样本
# ============================================================
def predict_one(text: str) -> str:
    """
    输入：一条文本
    输出：预测类别（中文label）
    """
    model.eval()

    # tokenizer 编码，直接返回 torch tensor
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    # 移到同一设备（GPU/CPU）
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 推理不计算梯度
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, num_labels]

    # 取最大概率的类别 id
    pred_id = int(torch.argmax(logits, dim=1).cpu().item())

    # 数字标签 -> 中文标签
    pred_label = lbl.inverse_transform([pred_id])[0]
    return pred_label


# ---- 验证样本 1：随机指令文本（不属于新闻分类，预测只是“硬归类”）----
sample1 = "下周一上午十点帮我定个闹钟"
print("\nsample:", sample1)
print("pred:", predict_one(sample1))

# ---- 验证样本 2：更像新闻的句子（更适合验证分类效果）----
sample2 = "国家队在决赛中以2比1逆转夺冠，球迷沸腾。"
print("\nsample:", sample2)
print("pred:", predict_one(sample2))
