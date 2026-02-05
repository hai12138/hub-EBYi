# 使用 GRU 代替 LSTM 实现05_LSTM文本分类.py
# 使用rnn/ lstm / gru 分别代替原始lstm，进行实验，对比精度

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
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

# ---------以上数据处理部分不变----------

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

# ---------以上自定义数据集部分不变--------只修改一个名字更为合理------
# CharLSTMDataset-》CharDataset

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

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out
# ———————————— 以上LSTM模型定义不变 ————————————
# ———————————— 增加以下RNN、GRU模型即可 ——————————
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        rnn_out, h_n = self.rnn(embedded)

        # batch size * output_dim
        out = self.fc(h_n.squeeze(0))
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        gru_out, h_n = self.gru(embedded)

        # batch size * output_dim
        out = self.fc(h_n.squeeze(0))
        return out
#——————————————————————————————————————————————————————————————————————————————
# lstm_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
# rnn_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
# gru_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

num_epochs = 4
for i in range(3):
    if i == 0:  # lstm
        print("——————开始训练lstm——————\n")
        for epoch in range(num_epochs):
            lstm_model.train()
            running_loss = 0.0
            for idx, (inputs, labels) in enumerate(dataloader):
                lstm_optimizer.zero_grad()
                outputs = lstm_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                lstm_optimizer.step()
                running_loss += loss.item()
                if idx % 50 == 0:
                    print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    elif i == 1: # RNN
        print("——————开始训练RNN——————\n")
        for epoch in range(num_epochs):
            rnn_model.train()
            running_loss = 0.0
            for idx, (inputs, labels) in enumerate(dataloader):
                rnn_optimizer.zero_grad()
                outputs = rnn_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                rnn_optimizer.step()
                running_loss += loss.item()
                if idx % 50 == 0:
                    print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    elif i == 2: # GRU
        print("——————开始训练gru——————\n")
        for epoch in range(num_epochs):
            gru_model.train()
            running_loss = 0.0
            for idx, (inputs, labels) in enumerate(dataloader):
                gru_optimizer.zero_grad()
                outputs = gru_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                gru_optimizer.step()
                running_loss += loss.item()
                if idx % 50 == 0:
                    print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

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
predicted_class = classify_text(new_text, lstm_model, char_to_index, max_len, index_to_label)
print(f"使用lstm | 输入 '{new_text}' 预测为: '{predicted_class}'")

predicted_class = classify_text(new_text, rnn_model, char_to_index, max_len, index_to_label)
print(f"使用rnn | 输入 '{new_text}' 预测为: '{predicted_class}'")

predicted_class = classify_text(new_text, gru_model, char_to_index, max_len, index_to_label)
print(f"使用gru | 输入 '{new_text}' 预测为: '{predicted_class}'")


new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, lstm_model, char_to_index, max_len, index_to_label)
print(f"使用lstm | 输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

predicted_class_2 = classify_text(new_text_2, rnn_model, char_to_index, max_len, index_to_label)
print(f"使用rnn | 输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

predicted_class_2 = classify_text(new_text_2, gru_model, char_to_index, max_len, index_to_label)
print(f"使用gru | 输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
