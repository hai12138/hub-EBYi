import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week02/dataset.csv", sep="\t", header=None)
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
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(model, dataloader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return loss_history


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

output_dim = len(label_to_index)

# 单层模型 (512 -> 128 -> output)
model_1 = SimpleClassifier(vocab_size, [128], output_dim)
print("\n=== 模型1: 单隐藏层 (128节点) ===")
losses_1 = train_model(model_1, dataloader)

# 双层模型 (512 -> 256 -> 128 -> output)
model_2 = SimpleClassifier(vocab_size, [256, 128], output_dim)
print("\n=== 模型2: 双隐藏层 (256->128节点) ===")
losses_2 = train_model(model_2, dataloader)

# 三层模型 (512 -> 512 -> 256 -> 128 -> output)
model_3 = SimpleClassifier(vocab_size, [512, 256, 128], output_dim)
print("\n=== 模型3: 三隐藏层 (512->256->128节点) ===")
losses_3 = train_model(model_3, dataloader)

# 更宽的单层模型 (512 -> 512 -> output)
model_4 = SimpleClassifier(vocab_size, [512], output_dim)
print("\n=== 模型4: 单隐藏层 (512节点) ===")
losses_4 = train_model(model_4, dataloader)

# 绘制损失比较图
plt.figure(figsize=(12, 8))
epochs = range(1, len(losses_1) + 1)
plt.plot(epochs, losses_1, label='模型1: [128]', marker='o')
plt.plot(epochs, losses_2, label='模型2: [256, 128]', marker='s')
plt.plot(epochs, losses_3, label='模型3: [512, 256, 128]', marker='^')
plt.plot(epochs, losses_4, label='模型4: [512]', marker='d')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型结构的训练损失对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# 验证模型复杂度统计
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"\n参数数量对比:")
print(f"模型1 (单层128): {count_parameters(model_1):,} 参数")
print(f"模型2 (双层256->128): {count_parameters(model_2):,} 参数")
print(f"模型3 (三层512->256->128): {count_parameters(model_3):,} 参数")
print(f"模型4 (单层512): {count_parameters(model_4):,} 参数")


# 最终测试
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
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


index_to_label = {i: label for label, i in label_to_index.items()}

test_texts = ["帮我导航到北京", "查询明天北京的天气"]

for test_text in test_texts:
    pred1 = classify_text(test_text, model_1, char_to_index, vocab_size, max_len, index_to_label)
    pred2 = classify_text(test_text, model_2, char_to_index, vocab_size, max_len, index_to_label)
    pred3 = classify_text(test_text, model_3, char_to_index, vocab_size, max_len, index_to_label)
    pred4 = classify_text(test_text, model_4, char_to_index, vocab_size, max_len, index_to_label)

    print(f"\n输入: '{test_text}'")
    print(f"  模型1预测: {pred1}")
    print(f"  模型2预测: {pred2}")
    print(f"  模型3预测: {pred3}")
    print(f"  模型4预测: {pred4}")
