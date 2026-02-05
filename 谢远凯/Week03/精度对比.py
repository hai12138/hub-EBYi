import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===================== 1. 数据加载与预处理（与原始代码一致）=====================
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签映射：字符串->数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
# 字符映射：字符->数字（含<pad>占位符）
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40  # 文本最大长度（截断/补零）


# ===================== 2. 自定义数据集（与原始代码一致）=====================
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
        # 截断+补零，统一长度
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# ===================== 3. 统一循环神经网络分类器（支持RNN/LSTM/GRU）=====================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='lstm'):
        """
        统一RNN分类器，支持RNN/LSTM/GRU
        :param vocab_size: 字符表大小
        :param embedding_dim: 嵌入层维度
        :param hidden_dim: 循环层隐藏维度
        :param output_dim: 输出类别数
        :param rnn_type: 循环层类型，可选['rnn', 'lstm', 'gru']
        """
        super(RNNClassifier, self).__init__()
        self.rnn_type = rnn_type.lower()
        # 嵌入层：字符索引->稠密向量（可训练）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 核心循环层：根据rnn_type选择，保持batch_first=True（输入格式：[batch, seq_len, dim]）
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"rnn_type仅支持['rnn','lstm','gru']，当前输入：{rnn_type}")
        # 全连接层：循环层输出->类别概率
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播：x -> 嵌入 -> 循环层 -> 隐藏状态 -> 类别输出
        :param x: 输入张量，形状[batch_size, seq_len]
        :return: 类别预测值，形状[batch_size, output_dim]
        """
        # 嵌入层：[batch, seq_len] -> [batch, seq_len, embedding_dim]
        embedded = self.embedding(x)
        # 循环层前向传播
        if self.rnn_type == 'lstm':
            # LSTM返回：输出序列+（隐藏状态，细胞状态），取最后一步隐藏状态
            rnn_out, (hidden, cell) = self.rnn(embedded)
        else:
            # RNN/GRU返回：输出序列+隐藏状态，直接取隐藏状态
            rnn_out, hidden = self.rnn(embedded)
        # 隐藏状态形状：[1, batch, hidden_dim] -> 挤压维度后[batch, hidden_dim]
        hidden = hidden.squeeze(0)
        # 全连接层：[batch, hidden_dim] -> [batch, output_dim]
        out = self.fc(hidden)
        return out


# ===================== 4. 训练与评估函数（含精度计算）=====================
def train_and_evaluate(model, dataloader, criterion, optimizer, num_epochs, model_name):
    """
    模型训练+精度评估，打印每轮Loss和整体精度
    :param model: 初始化的模型
    :param dataloader: 数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练轮数
    :param model_name: 模型名称（用于打印日志）
    :return: 训练完成的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"\n===== 开始训练{model_name}模型（设备：{device}）=====")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0  # 正确预测数
        total = 0  # 总样本数

        for idx, (inputs, labels) in enumerate(dataloader):
            # 数据移至设备
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播+参数更新
            loss.backward()
            optimizer.step()
            # 累计损失
            running_loss += loss.item()
            # 计算批次精度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 打印批次日志
            if idx % 50 == 0:
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                print(f"{model_name} | Batch {idx} | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.4f}")

        # 计算本轮整体Loss和精度
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(
            f"{model_name} | Epoch [{epoch + 1}/{num_epochs}] | Avg Loss: {epoch_loss:.4f} | Total Acc: {epoch_acc:.4f}")

    print(f"===== {model_name}模型训练完成 =====")
    return model


# ===================== 5. 预测函数（与原始逻辑一致，适配统一模型）=====================
def classify_text(text, model, char_to_index, max_len, index_to_label):
    """
    单文本分类预测
    :param text: 输入文本
    :param model: 训练好的模型
    :param char_to_index: 字符->索引映射
    :param max_len: 文本最大长度
    :param index_to_label: 索引->标签映射
    :return: 预测标签
    """
    device = next(model.parameters()).device  # 获取模型所在设备
    # 文本预处理：截断+补零
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    # 转换为张量并增加batch维度：[seq_len] -> [1, seq_len]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    # 模型评估模式，关闭梯度计算
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    # 取概率最大的类别
    _, predicted_index = torch.max(output, 1)
    predicted_label = index_to_label[predicted_index.item()]
    return predicted_label


# ===================== 6. 主程序：初始化+训练+对比=====================
if __name__ == "__main__":
    # 初始化数据集和数据加载器
    dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 超参数（三种模型完全一致，保证对比公平）
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)
    num_epochs = 4
    lr = 0.001

    # 标签反向映射：索引->字符串
    index_to_label = {i: label for label, i in label_to_index.items()}

    # 待测试的模型类型列表
    rnn_types = ['rnn', 'lstm', 'gru']
    trained_models = {}  # 保存训练好的模型

    # 循环训练三种模型
    for rnn_type in rnn_types:
        # 初始化模型、损失函数、优化器
        model = RNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rnn_type=rnn_type
        )
        criterion = nn.CrossEntropyLoss()  # 分类任务默认损失
        optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器

        # 训练并评估模型
        trained_model = train_and_evaluate(model, dataloader, criterion, optimizer, num_epochs, rnn_type.upper())
        trained_models[rnn_type] = trained_model

    # ===================== 7. 模型预测对比 =====================
    test_texts = [
        "帮我导航到北京",
        "查询明天北京的天气"
    ]
    print("\n===== 三种模型预测结果对比 =====")
    for text in test_texts:
        print(f"\n输入文本：{text}")
        for rnn_type in rnn_types:
            pred_label = classify_text(
                text=text,
                model=trained_models[rnn_type],
                char_to_index=char_to_index,
                max_len=max_len,
                index_to_label=index_to_label
            )
            print(f"  {rnn_type.upper()} 预测：{pred_label}")
