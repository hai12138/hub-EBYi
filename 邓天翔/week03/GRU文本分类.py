import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import embedding
from torch.utils.data import Dataset, DataLoader

# 读取原始数据集并编码
dataset = pd.read_csv("../week1/dataset.csv", sep="\t", header=None)
# 文本列
texts = dataset[0].tolist()
# 标签列
string_labels = dataset[1].tolist()

# 将标签映射为数字
label_to_index = {label:i for i,label in enumerate(set(string_labels))}
# 获取标签映射的数字列表
numeric_labels = [label_to_index[label] for label in string_labels]

# 将文本列字符映射为数字
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 倒转数字字符映射
index_to_char = {i : char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# 最大的文本输入长度
max_len = 40

# 自定义数据集
class CharGRUDataset(Dataset):
#     初始化
    def __init__(self, texts, labels, char_to_index, max_len):
#         文本输入
        self.texts = texts
# 文本对应的标签
        self.labels = torch.tensor(labels, dtype=torch.long)
# 字符到索引的映射
        self.char_to_index = char_to_index
# 文本最大输入长度
        self.max_len = max_len

# 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

#     获取单个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
#         取长补短
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 循环层
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
#        全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
# batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

# batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        gru_out, hidden_state = self.gru(embedded)
# batch size * hidden_dim
        out = self.fc(hidden_state.squeeze(0))
        return out

# 训练与预测过程
gru_dataset = CharGRUDataset(texts, numeric_labels, char_to_index, max_len)
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

# 定义模型
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 4
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    # 初始化损失
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        # 清除梯度
        optimizer.zero_grad()
#         向前传播
        outputs = model(inputs)
#         计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
#         损失累加
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch loss：{loss.item()}")
    # 打印每个epoch的平均损失
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

# 分类函数
def classify_text_gru(texts, model, char_to_index, max_len, index_to_label):
#     将文本取长补短
    indices = [char_to_index.get(char, 0) for char in texts[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # 切换为评估模式
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        output = model(input_tensor)

    # 从输出中找出评分最高的标签
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]
    return predicted_label

# 倒转标签映射
index_to_label = {i: label for label, i in label_to_index.items()}
new_text = "帮我导航到北京"
predicted_label = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_label}'")

new_text_2 = "查询明天北京的天气"
predicted_label_2 = classify_text_gru(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_label_2}'")

























