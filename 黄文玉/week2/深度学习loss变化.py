import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # 仅新增用于可视化对比

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
            tokenized = [self.char_to_index.get(char, 0) for char in text[:max_len]]
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MultiLayerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):

        super(MultiLayerClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        # 构建隐藏层
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        # 构建输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)


model_configs = {
    "原模型(2层-128节点)": (SimpleClassifier, [vocab_size, 128, len(label_to_index)]),
    "1层-64节点": (MultiLayerClassifier, [vocab_size, [64], len(label_to_index)]),
    "2层-64-32节点": (MultiLayerClassifier, [vocab_size, [64, 32], len(label_to_index)]),
    "3层-128-64-32节点": (MultiLayerClassifier, [vocab_size, [128, 64, 32], len(label_to_index)]),
    "2层-256节点": (MultiLayerClassifier, [vocab_size, [256], len(label_to_index)])
}


def train_single_model(model, dataloader, num_epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []  # 记录每轮loss

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
            if idx % 50 == 0 and epoch == 0:  # 仅第一轮打印batch loss，避免冗余
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"【{model_name}】Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return loss_history


loss_records = {}
num_epochs = 10
print("===== 开始训练不同层数/节点数的模型 =====")
for model_name, (model_cls, params) in model_configs.items():
    print(f"\n--- 训练 {model_name} ---")
    model = model_cls(*params)
    loss_hist = train_single_model(model, dataloader, num_epochs)
    loss_records[model_name] = loss_hist

# 新增：可视化不同模型的loss变化（可选，便于对比）
plt.figure(figsize=(10, 6))
for name, loss_hist in loss_records.items():
    plt.plot(range(1, num_epochs + 1), loss_hist, label=name)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("不同层数/节点数模型的Loss对比")
plt.legend()
plt.grid(True)
plt.show()


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
original_model = SimpleClassifier(vocab_size, 128, len(label_to_index))
_ = train_single_model(original_model, dataloader, num_epochs=10)

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, original_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"\n输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, original_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

#
# # 原有预测输出（完全不变）
# -- 训练 3层-128-64-32节点 ---
# Batch 个数 0, 当前Batch Loss: 2.4946107864379883
# Batch 个数 50, 当前Batch Loss: 2.4342117309570312
# Batch 个数 100, 当前Batch Loss: 2.464979887008667
# Batch 个数 150, 当前Batch Loss: 2.4444148540496826
# Batch 个数 200, 当前Batch Loss: 2.4455466270446777
# Batch 个数 250, 当前Batch Loss: 2.4205517768859863
# Batch 个数 300, 当前Batch Loss: 2.438310384750366
# Batch 个数 350, 当前Batch Loss: 2.3994762897491455
# 【3层-128-64-32节点】Epoch [1/10], Loss: 2.4442
# 【3层-128-64-32节点】Epoch [2/10], Loss: 2.4016
# 【3层-128-64-32节点】Epoch [3/10], Loss: 2.3763
# 【3层-128-64-32节点】Epoch [4/10], Loss: 2.3602
# 【3层-128-64-32节点】Epoch [5/10], Loss: 2.3484
# 【3层-128-64-32节点】Epoch [6/10], Loss: 2.3375
# 【3层-128-64-32节点】Epoch [7/10], Loss: 2.3235
# 【3层-128-64-32节点】Epoch [8/10], Loss: 2.2935
# 【3层-128-64-32节点】Epoch [9/10], Loss: 2.2156
# 【3层-128-64-32节点】Epoch [10/10], Loss: 1.9665
#
# --- 训练 2层-256节点 ---
# Batch 个数 0, 当前Batch Loss: 2.4813547134399414
# Batch 个数 50, 当前Batch Loss: 2.4648234844207764
# Batch 个数 100, 当前Batch Loss: 2.450039863586426
# Batch 个数 150, 当前Batch Loss: 2.4405062198638916
# Batch 个数 200, 当前Batch Loss: 2.416746139526367
# Batch 个数 250, 当前Batch Loss: 2.3973937034606934
# Batch 个数 300, 当前Batch Loss: 2.36240816116333
# Batch 个数 350, 当前Batch Loss: 2.3308169841766357
# 【2层-256节点】Epoch [1/10], Loss: 2.4161
# 【2层-256节点】Epoch [2/10], Loss: 2.2352
# 【2层-256节点】Epoch [3/10], Loss: 1.9395
# 【2层-256节点】Epoch [4/10], Loss: 1.5625
# 【2层-256节点】Epoch [5/10], Loss: 1.2273
# 【2层-256节点】Epoch [6/10], Loss: 0.9853
# 【2层-256节点】Epoch [7/10], Loss: 0.8282
# 【2层-256节点】Epoch [8/10], Loss: 0.7221
# 【2层-256节点】Epoch [9/10], Loss: 0.6436
# 【2层-256节点】Epoch [10/10], Loss: 0.5865
