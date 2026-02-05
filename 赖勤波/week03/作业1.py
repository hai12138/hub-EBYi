"""
运行结果，RNN的性能相对来说差一点，在同样层数、同样embedding维度、同样hidden-dim的情况下
lstm-输入 '帮我导航到北京' 预测为: 'Travel-Query'
rnn-输入 '帮我导航到北京' 预测为: 'Music-Play'
gru-输入 '帮我导航到北京' 预测为: 'Travel-Query'
lstm-输入 '查询明天北京的天气' 预测为: 'Weather-Query'
rnn-输入 '查询明天北京的天气' 预测为: 'Weather-Query'
gru-输入 '查询明天北京的天气' 预测为: 'Weather-Query'
"""


import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# set_seed(42)

dataset = pd.read_csv("../../第1周：课程介绍与大模型基础/Week01/dataset.csv", sep="\t", header=None)
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

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # lstm_out: batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim   记录最新层的所有seq_len对应的隐藏状态
        # hidden_state: [num_layer*direction, batch size, hidden_dim]  只记录每一层每个方向的最后的隐藏状态
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.RNN = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.RNN(embedded)
        out = self.fc(hidden.squeeze(0))
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        out = self.fc(hidden.squeeze(0))
        return out


print("---LSTM Training and Prediction ---\n")
# ---lstm Training and Prediction ---
lstm_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader_lstm = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model_lstm = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    model_lstm.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader_lstm):
        optimizer.zero_grad()
        outputs = model_lstm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader_lstm):.4f}")

print("---RNN Training and Prediction ---\n")
# ---RNN Training and Prediction ---
rnn_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader_rnn = DataLoader(rnn_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model_rnn = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_rnn.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    model_rnn.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader_rnn):
        optimizer.zero_grad()
        outputs = model_rnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader_rnn):.4f}")
print("---Gru Training and Prediction ---\n")
# ---Gru Training and Prediction ---
gru_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader_gru = DataLoader(gru_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model_gru = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_gru.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    model_gru.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader_gru):
        optimizer.zero_grad()
        outputs = model_gru(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader_gru):.4f}")

def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class_lstm = classify_text(new_text, model_lstm, char_to_index, max_len, index_to_label)
print(f"lstm-输入 '{new_text}' 预测为: '{predicted_class_lstm}'")
predicted_class_rnn = classify_text(new_text, model_rnn, char_to_index, max_len, index_to_label)
print(f"rnn-输入 '{new_text}' 预测为: '{predicted_class_rnn}'")
predicted_class_gru = classify_text(new_text, model_gru, char_to_index, max_len, index_to_label)
print(f"gru-输入 '{new_text}' 预测为: '{predicted_class_gru}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2_lstm = classify_text(new_text_2, model_lstm, char_to_index, max_len, index_to_label)
print(f"lstm-输入 '{new_text_2}' 预测为: '{predicted_class_2_lstm}'")
predicted_class_2_rnn = classify_text(new_text_2, model_rnn, char_to_index, max_len, index_to_label)
print(f"rnn-输入 '{new_text_2}' 预测为: '{predicted_class_2_rnn}'")
predicted_class_2_gru = classify_text(new_text_2, model_gru, char_to_index, max_len, index_to_label)
print(f"gru-输入 '{new_text_2}' 预测为: '{predicted_class_2_gru}'")
