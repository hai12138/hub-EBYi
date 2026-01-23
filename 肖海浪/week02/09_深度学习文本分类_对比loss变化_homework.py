import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import random
import numpy as np

# 固定随机种子，便于不同结构之间做可重复对比
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../lang_week02/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        """中间层分类
        hidden_dims;list[int]
        """

        # 层初始化
        super(SimpleClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据


def train_and_get_losses(model, dataloader, num_epochs=10, lr=0.01, optimizer_name="SGD"):
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.upper() == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] AvgLoss: {avg_loss:.4f}")

    return epoch_losses


# =========================
# 多组模型结构对比实验
# =========================

# 训练超参数（可以随时调整）
num_epochs = 10
lr = 0.01

# 先用 SGD 跑一遍，再切 Adam 对比收敛差异
optimizer_name = "SGD"  # 可选："SGD" 或 "Adam"

# output_dim = 类别数
output_dim = len(label_to_index)

# 配置组：层数 = 列表长度；每层节点数 = 列表里的数字
# 可以自行增删，让对比更聚焦
configs = {
    "1层-64": [64],
    "1层-128": [128],
    "1层-256": [256],
    "2层-256-128": [256, 128],
    "3层-256-128-64": [256, 128, 64],
}

# 保存每个配置的 loss 曲线
all_losses = {}

for name, hidden_dim in configs.items():
    print("\n" + "=" * 70)
    print(f"Config: {name} | hidden_dims={hidden_dim}")

    # 每个配置都重新初始化一个模型，确保对比的是“结构差异”
    model = SimpleClassifier(input_dim=vocab_size, hidden_dim=hidden_dim, output_dim=output_dim)

    # 训练并得到每轮 AvgLoss
    losses = train_and_get_losses(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        lr=lr,
        optimizer_name=optimizer_name
    )

    all_losses[name] = losses

# 汇总：对比每组最后一个 epoch 的 AvgLoss
print("\n" + "#" * 70)
print("Loss对比（每组最后一个 epoch 的 AvgLoss，越小表示训练集拟合越充分）：")
for name, losses in all_losses.items():
    print(f"{name:>18}: {losses[-1]:.4f}")



def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
