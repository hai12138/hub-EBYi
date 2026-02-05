# m第二周作业
# 1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim): # 层的个数 和 验证集精度
        super(SimpleClassifier, self).__init__()
        layers = []
        prev_dim = input_dim # prev_dim：每一个全连接层的前边神经元维度
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim # 关键一步！
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        # 组合所有层
        self.mlp_model = nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp_model(x)



def train_loss(hidden_dims, num_epochs=10, save_path="./weights"):
    """

    Args:
        hidden_dims:
        num_epochs:

    Returns:
        每轮的损失
    """
    # 创建模型
    output_dim = len(label_to_index)
    model = SimpleClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss() # 多分类，交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epoch_losses = []

    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
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
        # 计算每轮loss并进行记录
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"hidden_dims = {hidden_dims} | Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    save_path = os.path.join(save_path, f"/{hidden_dims}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型配置 {hidden_dims} 权重已保存至 {save_path}")

    return epoch_losses



def classify_text(text, model_config, char_to_index, vocab_size, max_len, index_to_label, model_path="./weights"):
    # 根据传入的模型配置，重构模型
    output_dim = len(index_to_label)
    model = SimpleClassifier(vocab_size, model_config, output_dim)
    model_path = os.path.join(model_path, f"/{model_config}.pth")
    model.load_state_dict(torch.load(model_path))

    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

if __name__ == '__main__':
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


    char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)  # 读取批量数据集 -》 batch数据

    # 自定义对比的模型配置
    zdy_configs = [
        [64],  # 配置1：1层隐藏层，64个节点
        [128],  # 配置2：1层隐藏层，128个节点（原配置）
        [256],  # 配置3：1层隐藏层，256个节点
        [128, 64],  # 配置4：2层隐藏层，128→64节点
        [256, 128, 64]  # 配置5：3层隐藏层，256→128→64节点
    ]

    loss_results = {} # 字典类型
    for config in zdy_configs:
        loss_results[str(config)] = train_loss(config)

    plt.figure(figsize=(10, 6))
    for config_str, losses in loss_results.items():
        plt.plot(range(1, 11), losses, label=f"模型配置: {config_str}")

    plt.xlabel("训练轮数（Epoch）")
    plt.ylabel("Loss（损失值）")
    plt.title("不同模型层数/节点数的Loss变化对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("loss_comparison.png")  # 保存图片
    plt.show()

    # 5. 打印最终loss对比（方便看数值）
    print("\n===== 最终Loss对比 =====")
    for config_str, losses in loss_results.items():
        print(f"配置{config_str}: 初始Loss={losses[0]:.4f}, 最终Loss={losses[-1]:.4f}")

# 进行预测
index_to_label = {i: label for label, i in label_to_index.items()}
best_config = min(loss_results.keys(), key=lambda x: loss_results[x][-1])
print(f"\n最佳配置：{best_config}")

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, eval(best_config), char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, eval(best_config), char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

