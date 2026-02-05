import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载和预处理
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


class CharDataset(Dataset):
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


# 通用循环神经网络分类器
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='rnn'):
        super(RNNClassifier, self).__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 根据类型选择不同的RNN层
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:  # 默认RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embedded = self.embedding(x)

        if self.rnn_type.lower() == 'lstm':
            rnn_out, (hidden, cell) = self.rnn(embedded)
            # 取LSTM最后一个时间步的隐藏状态
            out = self.fc(hidden.squeeze(0))
        else:
            rnn_out, hidden = self.rnn(embedded)
            # 取RNN/GRU最后一个时间步的隐藏状态
            out = self.fc(hidden.squeeze(0))

        return out


# 训练和评估函数
def train_and_evaluate_model(model, dataloader, criterion, optimizer, num_epochs, model_name):
    """训练模型并返回训练过程中的损失和准确率"""
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f'{model_name} Epoch {epoch + 1}/{num_epochs}')

        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
            })

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct_predictions / total_samples

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f'{model_name} Epoch [{epoch + 1}/{num_epochs}], '
              f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    return train_losses, train_accuracies


def evaluate_model(model, dataloader, model_name):
    """评估模型在测试集上的性能"""
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = 100 * correct_predictions / total_samples
    print(f'{model_name} 测试准确率: {accuracy:.2f}%')
    return accuracy


# 创建数据集和数据加载器
full_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)

# 分割训练集和测试集 (80% 训练, 20% 测试)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_epochs = 10

# 定义三种模型
models = {
    'RNN': RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 'rnn'),
    'LSTM': RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 'lstm'),
    'GRU': RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 'gru')
}

# 存储每种模型的训练结果
results = {}

# 训练和评估每种模型
for model_name, model in models.items():
    print(f'\n{"=" * 50}')
    print(f'开始训练 {model_name} 模型')
    print(f'{"=" * 50}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_losses, train_accuracies = train_and_evaluate_model(
        model, train_dataloader, criterion, optimizer, num_epochs, model_name
    )

    # 评估模型
    test_accuracy = evaluate_model(model, test_dataloader, model_name)

    # 保存结果
    results[model_name] = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracy': test_accuracy,
        'model': model
    }

# 绘制对比图表
plt.figure(figsize=(15, 5))

# 1. 训练损失对比
plt.subplot(1, 3, 1)
for model_name, result in results.items():
    plt.plot(result['train_losses'], label=model_name, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 训练准确率对比
plt.subplot(1, 3, 2)
for model_name, result in results.items():
    plt.plot(result['train_accuracies'], label=model_name, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('训练准确率对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. 测试准确率对比
plt.subplot(1, 3, 3)
model_names = list(results.keys())
test_accuracies = [results[name]['test_accuracy'] for name in model_names]
bars = plt.bar(model_names, test_accuracies, color=['#ff9999', '#66b3ff', '#99ff99'])
plt.xlabel('模型类型')
plt.ylabel('准确率 (%)')
plt.title('测试集准确率对比')

# 在柱状图上显示数值
for bar, accuracy in zip(bars, test_accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{accuracy:.2f}%', ha='center', va='bottom')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 打印最终结果对比
print('\n' + '=' * 60)
print('模型性能对比总结')
print('=' * 60)
for model_name, result in results.items():
    print(f'{model_name}:')
    print(f'  最终训练损失: {result["train_losses"][-1]:.4f}')
    print(f'  最终训练准确率: {result["train_accuracies"][-1]:.2f}%')
    print(f'  测试集准确率: {result["test_accuracy"]:.2f}%')
    print()

# 使用最佳模型进行预测
best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
best_model = results[best_model_name]['model']
print(f'最佳模型: {best_model_name} (测试准确率: {results[best_model_name]["test_accuracy"]:.2f}%)')


def classify_text(text, model, char_to_index, max_len, index_to_label, model_type='rnn'):
    """使用指定模型进行文本分类"""
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


# 创建索引到标签的映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试样例
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放周杰伦的歌",
    "今天天气怎么样"
]

print('\n模型预测结果:')
for text in test_texts:
    prediction = classify_text(text, best_model, char_to_index, max_len, index_to_label)
    print(f'输入: "{text}" -> 预测: "{prediction}"')
