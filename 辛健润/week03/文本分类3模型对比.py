
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据预处理
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
max_len = 40

class CharLSTMDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

#模型定义
#RNN
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, _) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# GRU
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# 训练与测试函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=4):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

#预测结果
def evaluate_model(model, test_texts, char_to_index, max_len, index_to_label):
    model.eval()
    results = []
    with torch.no_grad():
        for text in test_texts:
            indices = [char_to_index.get(char, 0) for char in text[:max_len]]
            indices += [0] * (max_len - len(indices))
            input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
            output = model(input_tensor)
            _, predicted_index = torch.max(output, 1)
            predicted_label = index_to_label[predicted_index.item()]
            results.append((text, predicted_label))
    return results

# 主程序入口
if __name__ == "__main__":
    # 加载数据
    lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
    dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

    # 超参数
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)

    # 模型列表
    models = {
        "RNN": RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        "LSTM": LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        "GRU": GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    }

    # 训练与评估
    criterion = nn.CrossEntropyLoss()
    test_texts = ["帮我导航到北京", "查询明天北京的天气"]
    index_to_label = {i: label for label, i in label_to_index.items()}

    for name, model in models.items():
        print(f"\n--- 训练 {name} 模型 ---")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, dataloader, criterion, optimizer)

        print(f"\n--- 测试 {name} 模型 ---")
        results = evaluate_model(model, test_texts, char_to_index, max_len, index_to_label)
        for text, pred in results:
            print(f"输入 '{text}' 预测为: '{pred}'")
