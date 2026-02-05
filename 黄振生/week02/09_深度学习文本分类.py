"""
hzs, 2026-01-22, 作业：深度学习分类

项目目标：
创建一个能够识别中文文本意图的分类器，如何判断“帮我导航到北京”是导航类还是其它类；
通过调整模型层数和节点数，比对模型的loss变化
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==================== 数据加载与预处理 ====================
def load_and_preprocess_data(file_path):
    # 读取数据
    dataset = pd.read_csv(file_path, sep="\t", header=None, names=['text', 'label'])

    # 提取文本和标签
    texts = dataset['text'].tolist()
    string_labels = dataset['label'].tolist()

    # 将类别转为数字
    label_to_index = { label: i for i, label in enumerate(set(string_labels))}
    numerical_labels = [ label_to_index[label] for label in string_labels]

    # 字符级词汇表构建
    char_to_index = { '<pad>': 0,  '<unk>': 1 } # 填充符，用0表示
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)

    index_to_char = {i: char for char, i in char_to_index.items()}
    vocab_size = len(char_to_index)

    print(f"数据集大小: {len(texts)}")
    print(f"词汇表大小: {vocab_size}")
    print(f"类别数量: {len(label_to_index)}")
    print(f"类别映射: {label_to_index}")

    return texts, numerical_labels, char_to_index, label_to_index, index_to_char

# ==================== 改进的数据集类 ====================
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len=40):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 将文本转换为字符索引序列
        sequence = []
        for char in text[:self.max_len]:
            sequence.append(self.char_to_index.get(char, 1))   # 未知字符用1表示

        # 填充到最大长度
        while len(sequence) < self.max_len:
            sequence.append(0)

        return torch.tensor(sequence), label

# ==================== 多层神经网络模型 ====================
class MultiLayerTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dims=[256, 128], num_classes=13, dropout_rate=0.3):
        """
        多层文本分类器

        Args:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dims: 隐藏层维度列表
            num_classes: 类别数量
            dropout_rate: dropout概率
        """
        super(MultiLayerTextClassifier, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 构建隐藏层
        layers = []
        input_dim = embed_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # 批归一化加速训练
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # 使用平均池化将变长序列转换为固定长度
        pooled = torch.mean(embedded, dim=1)  # (batch_size, embed_dim)

        output = self.network(pooled)
        return output

# ==================== 训练函数 ====================
def train_model(model, dataloader, criterion, optimizer, num_epochs=15):
    """训练模型"""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# ==================== 预测函数 ====================
def predict_text(text, model, char_to_index, max_len=50, index_to_label=None):
    """预测单个文本的类别"""
    # 文本预处理
    sequence = []
    for char in text[:max_len]:
        sequence.append(char_to_index.get(char, 1))  # 未知字符

    while len(sequence) < max_len:
        sequence.append(0)  # 填充

    input_tensor = torch.tensor([sequence], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    predicted_label = index_to_label[predicted_idx] if index_to_label else f"Class_{predicted_idx}"

    return predicted_label, confidence

# ==================== 主程序 ====================
def main():
    # 加载数据
    file_path = "../week01/dataset.csv"
    texts, numerical_labels, char_to_index, label_to_index, index_to_char = load_and_preprocess_data(file_path)

    # 创建数据集和数据加载器
    dataset = TextClassificationDataset(texts, numerical_labels, char_to_index, max_len=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型参数配置
    MODEL_CONFIG = {
        'vocab_size': len(char_to_index),
        'embed_dim': 128,
        'hidden_dims': [256, 128, 64],  # 三层隐藏层
        'num_classes': len(label_to_index),
        'dropout_rate': 0.3
    }

    # 创建模型
    model = MultiLayerTextClassifier(**MODEL_CONFIG)

    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器通常效果更好

    # 训练模型
    print("开始训练模型...")
    train_model(model, dataloader, criterion, optimizer, num_epochs=15)

    # 测试预测
    print("\n测试预测功能:")
    index_to_label = {v: k for k, v in label_to_index.items()}

    test_texts = [
        "帮我导航到北京",
        "查询明天北京的天气",
        "播放周杰伦的歌",
        "设置明天早上8点的闹钟"
    ]

    for text in test_texts:
        predicted_label, confidence = predict_text(text, model, char_to_index, 50, index_to_label)
        print(f"输入: '{text}' -> 预测: {predicted_label} (置信度: {confidence:.2f})")

# 程序入口
if __name__ == "__main__":
    main()

