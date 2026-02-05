import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # 新增：用于绘图

# ... (数据加载和预处理部分与您的原始代码完全一致) ...
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


# ... (CharBoWDataset 类定义与您的原始代码完全一致) ...
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


# --- 主要修改点 1：重构模型类，使其支持灵活定义层数 ---
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        input_dim: 输入维度（词汇表大小）
        hidden_dims: 一个列表，定义每个隐藏层的神经元数。
                     例如：[128] 表示单隐藏层（128个神经元）；
                           [128, 64] 表示双隐藏层（第一层128，第二层64）
        output_dim: 输出维度（类别数）
        """
        super(FlexibleClassifier, self).__init__()
        layers = []

        # 构建隐藏层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 添加非线性激活函数
            prev_dim = hidden_dim

        # 添加输出层（无激活函数，因为CrossEntropyLoss包含Softmax）
        layers.append(nn.Linear(prev_dim, output_dim))

        # 用Sequential将所有的层组合成一个完整的模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- 主要修改点 2：创建数据集和数据加载器 ---
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

output_dim = len(label_to_index)

# --- 主要修改点 3：定义要对比的模型架构 ---
# 实验配置：键为模型描述，值为隐藏层维度列表
experiments = {
    "1 Hidden Layer (128)": [128],  # 单隐藏层，128个神经元
    "2 Hidden Layers (128, 64)": [128, 64]  # 双隐藏层，第一层128，第二层64
}

# 存储每个模型的训练损失历史
results = {}

# --- 主要修改点 4：循环训练每一个模型 ---
for exp_name, hidden_dims in experiments.items():
    print(f"\n=== 开始训练模型: {exp_name} ===")

    # 初始化模型、损失函数、优化器
    model = FlexibleClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 用于记录本模型每个epoch的平均损失
    epoch_losses = []

    num_epochs = 10
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

        # 计算并记录本epoch的平均损失
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 训练完成后，将本模型的损失记录存入results字典
    results[exp_name] = epoch_losses

# --- 主要修改点 5：训练完成后，在同一张图上绘制所有模型的损失曲线 ---
plt.figure(figsize=(10, 6))

for exp_name, losses in results.items():
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label=exp_name)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison: Different Model Architectures')
plt.legend()  # 显示图例
plt.grid(True)
plt.savefig('loss_comparison.png')  # 保存图片
plt.show()  # 显示图片`

print("\n=== 训练和对比完成！图片已保存为 'loss_comparison.png'。 ===")
