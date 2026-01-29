import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 当前日期时间戳
start_time = pd.Timestamp.now()
# 读取数据集文件，解析文本和标签
dataset = pd.read_csv("../week1/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 将字符串标签转换为数值标签
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符到索引的映射字典
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# max length 最大输入的文本长度
max_len = 40


class CharRNNDataset(Dataset):
    """
    字符级RNN数据集类

    该类用于将文本数据转换为适合RNN模型训练的格式，包括字符编码、填充等预处理操作。
    """

    def __init__(self, texts, labels, char_to_index, max_len):
        """
        初始化CharRNNDataset实例

        Args:
            texts (list): 文本列表
            labels (list): 标签列表
            char_to_index (dict): 字符到索引的映射字典
            max_len (int): 文本最大长度
        """
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        """
        获取数据集样本数量

        Returns:
            int: 数据集中样本的数量
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取指定索引位置的样本

        Args:
            idx (int): 样本索引

        Returns:
            tuple: 包含编码后的文本张量和标签张量的元组
        """
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


class RNNClassifier(nn.Module):
    """
    基于RNN的文本分类器

    该模型使用字符级嵌入和RNN层进行文本分类任务。
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """
        初始化RNN分类器

        Args:
            vocab_size (int): 词汇表大小
            embedding_dim (int): 嵌入层维度
            hidden_dim (int): RNN隐藏层维度
            output_dim (int): 输出层维度（类别数量）
        """
        super(RNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        定义前向传播过程

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, seq_length)

        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, output_dim)
        """
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


# 训练和预测
rnn_dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(rnn_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text_rnn(text, model, char_to_index, max_len, index_to_label):
    """
    使用训练好的RNN模型对文本进行分类预测

    Args:
        text (str): 待分类的文本
        model (RNNClassifier): 训练好的RNN模型
        char_to_index (dict): 字符到索引的映射字典
        max_len (int): 文本最大长度
        index_to_label (dict): 索引到标签的映射字典

    Returns:
        str: 预测的文本类别标签
    """
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
predicted_class = classify_text_rnn(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' RNN预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_rnn(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' RNN预测为: '{predicted_class_2}'")
print("-----------------------------------------------------------------------------------------------------------------")
# 输出总共的用时
time = pd.Timestamp.now()
print("总共用时：", time - start_time)


# Epoch [4/4], Loss: 2.3548
# 输入 '帮我导航到北京' RNN预测为: 'FilmTele-Play'
# 输入 '查询明天北京的天气' RNN预测为: 'FilmTele-Play'
# -----------------------------------------------------------------------------------------------------------------
# 总共用时： 0 days 00:00:20.235372
