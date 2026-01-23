import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

# 获取数据集
datasets = pandas.read_csv("/Users/guozhijia/Documents/八斗/第一周：课程介绍与大模型基础/Week01/dataset.csv", sep="\t", header=None)
texts = datasets[0].tolist()
labels_str = datasets[1].tolist()

# label转化为数字
labels_map = {label: i for i, label in enumerate(set(labels_str))}
labels_num = [labels_map[label] for label in labels_str]

# 构建字符表和映射表
char_to_index = {'<pad>': 0}

for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}

# 构建自己的数据集加载器，需要重写__init__、_create_bow_vectors、__len__、__getitem__
class CharBowdataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = len(char_to_index)
        self.bow_vectors = self._create_bow_vectors()
    
    def _create_bow_vectors(self):
        char_index_list = []
        for text in self.texts:
            char_index = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            char_index.append([0] * (self.max_len - len(char_index)))
            char_index_list.append(char_index)
        
        bow_vectors = []
        for char_index in char_index_list:
            bow_vector = torch.zeros(self.vocab_size)
            for index in char_index:
                bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

# 构建模型,继承torch.nn.Module，重写__init__、forward
# 两层的网络
class ClassifyForTwoLinear(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, output_dim):
        super(ClassifyForTwoLinear, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim1, output_dim)
    
    def forward(self, input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        return output

# 三层的网络
class ClassifyForThreeLinear(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, output_dim):
        super(ClassifyForThreeLinear, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim1)
        self.fc3 = torch.nn.Linear(hidden_dim1, output_dim)
    
    def forward(self, input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output

# 初始化训练
charBowdataset = CharBowdataset(texts, labels_num, char_to_index, 40)
dataLoader = DataLoader(charBowdataset, batch_size=32, shuffle=True)

hidden_dim1 = 128
hidden_dim2 = 256
output_dim = len(labels_map)
# 2层网络
classify1 = ClassifyForTwoLinear(len(char_to_index), hidden_dim1, output_dim)
# 3层网络
classify2 = ClassifyForThreeLinear(len(char_to_index), hidden_dim1, output_dim)
# 2层网络，增大网络隐藏层的神经元
classify3 = ClassifyForTwoLinear(len(char_to_index), hidden_dim2, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(classify1.parameters(), lr=0.01)
optimizer2 = optim.Adam(classify2.parameters(), lr=0.01)
optimizer3 = optim.Adam(classify3.parameters(), lr=0.01)

num_epochs = 10
loss_history_1_layers = []
loss_history_2_layers = []
loss_history_3_layers = []

for epoch in range(num_epochs):
    classify1.train()
    classify2.train()
    classify3.train()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0

    for idx, (inputs, labels) in enumerate(dataLoader):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        output1 = classify1(inputs)
        output2 = classify2(inputs)
        output3 = classify3(inputs)

        loss1 = criterion(output1, labels)
        loss2 = criterion(output2, labels)
        loss3 = criterion(output3, labels)

        loss1.backward()
        loss2.backward()
        loss3.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        running_loss3 += loss3.item()
    
    epoch_loss1 = running_loss1 / len(dataLoader)
    epoch_loss2 = running_loss2 / len(dataLoader)
    epoch_loss3 = running_loss3 / len(dataLoader)
    
    loss_history_1_layers.append(epoch_loss1)
    loss_history_2_layers.append(epoch_loss2)
    loss_history_3_layers.append(epoch_loss3)
        
    print(f"2层网络的训练 - Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss1:.4f}")
    print(f"3层网络的训练 - Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss2:.4f}")
    print(f"改变节点的训练 - Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss3:.4f}")

# 绘制loss对比图
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_history_1_layers, label='2 layer Loss', marker='o')
plt.plot(range(1, num_epochs + 1), loss_history_2_layers, label='3 layer Loss', marker='s')
plt.plot(range(1, num_epochs + 1), loss_history_3_layers, label='2 layer 256 node Loss', marker='^')
plt.title('Loss comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

output_path = os.path.join(os.path.dirname(__file__), 'loss_comparison.png')
plt.savefig(output_path)
print(f"Loss对比图已保存至: {output_path}")
plt.show()

'''
loss结果：
    2层网络的训练 - Epoch [1/10], Loss: 0.4026
    3层网络的训练 - Epoch [1/10], Loss: 0.4581
    改变节点的训练 - Epoch [1/10], Loss: 0.4080
    2层网络的训练 - Epoch [2/10], Loss: 0.1091
    3层网络的训练 - Epoch [2/10], Loss: 0.1832
    改变节点的训练 - Epoch [2/10], Loss: 0.1119
    2层网络的训练 - Epoch [3/10], Loss: 0.0436
    3层网络的训练 - Epoch [3/10], Loss: 0.1136
    改变节点的训练 - Epoch [3/10], Loss: 0.0440
    2层网络的训练 - Epoch [4/10], Loss: 0.0267
    3层网络的训练 - Epoch [4/10], Loss: 0.0640
    改变节点的训练 - Epoch [4/10], Loss: 0.0303
    2层网络的训练 - Epoch [5/10], Loss: 0.0140
    3层网络的训练 - Epoch [5/10], Loss: 0.0442
    改变节点的训练 - Epoch [5/10], Loss: 0.0267
    2层网络的训练 - Epoch [6/10], Loss: 0.0102
    3层网络的训练 - Epoch [6/10], Loss: 0.0573
    改变节点的训练 - Epoch [6/10], Loss: 0.0227
    2层网络的训练 - Epoch [7/10], Loss: 0.0148
    3层网络的训练 - Epoch [7/10], Loss: 0.0588
    改变节点的训练 - Epoch [7/10], Loss: 0.0321
    2层网络的训练 - Epoch [8/10], Loss: 0.0332
    3层网络的训练 - Epoch [8/10], Loss: 0.0568
    改变节点的训练 - Epoch [8/10], Loss: 0.0237
    2层网络的训练 - Epoch [9/10], Loss: 0.0233
    3层网络的训练 - Epoch [9/10], Loss: 0.0470
    改变节点的训练 - Epoch [9/10], Loss: 0.0379
    2层网络的训练 - Epoch [10/10], Loss: 0.0161
    3层网络的训练 - Epoch [10/10], Loss: 0.0476
    改变节点的训练 - Epoch [10/10], Loss: 0.0451
'''