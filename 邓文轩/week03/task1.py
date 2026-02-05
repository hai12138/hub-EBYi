import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from re import X
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
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
class CharLSTMDataset(Dataset):
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

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) # (n,sen_len) -> (n,sen_len,embedding_dim)
        output,h = self.rnn(x)
        x = self.fc(h.squeeze(0)) # (n,embedding_dim) -> (n,output_dim)
        return x

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) # (n,sen_len) -> (n,sen_len,embedding_dim)
        output,h = self.gru(x)
        x = self.fc(h.squeeze(0)) # (n,embedding_dim) -> (n,output_dim)
        return x

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
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
    

# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)

# 划分训练集和验证集
train_size = int(0.8 * len(lstm_dataset))
val_size = len(lstm_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(lstm_dataset, [train_size, val_size])

# 创建训练集和验证集的 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

models = {
    "rnn":RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
    "lstm":LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
    "gru":GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
}

index_to_label = {i: label for label, i in label_to_index.items()}

models_acc = {}

def train(model,num_epochs,train_dataloader,val_dataloader,lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # 返回值
    best_acc = 0
    loss_change = []
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if idx % 10 == 0:
                # print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
                # loss_change.append(loss.item())

        train_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")
        loss_change.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_dataloader)
        val_accuracy = val_correct / val_total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        best_acc = max(best_acc, val_accuracy)

    return best_acc,loss_change


models_acc = {}
models_loss_change = {}
for model_name,model in models.items(): 
    print('='*20, f"开始训练并测试模型：{model_name}", '='*20)

    best_acc,loss_change = train(model,10,train_dataloader,val_dataloader,0.001)

    models_acc[model_name] = best_acc
    models_loss_change[model_name] = loss_change

    # 使用模型
    new_text = "帮我导航到东京"
    predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

    new_text_3 = "我想听摇滚"
    predicted_class_3 = classify_text_lstm(new_text_3, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text_3}' 预测为: '{predicted_class_3}'")


# 绘制损失变化图
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
for model_name, loss_change in models_loss_change.items():
    plt.plot(loss_change, label=model_name)
plt.title("Loss Change over Epochs")
plt.xlabel("Epoch")

plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
model_names = list(models_acc.keys())
acc_values = list(models_acc.values())
plt.bar(model_names, acc_values)
plt.title("Best Accuracy for Each Model")
plt.xlabel("Model")
plt.ylabel("Accuracy")
for i, v in enumerate(acc_values):
    plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("models_comparison.png")
plt.show()









