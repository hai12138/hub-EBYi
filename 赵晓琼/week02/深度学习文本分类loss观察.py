import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1.加载数据集
dataset = pd.read_csv('./dataset.csv', sep='\t', header=None)
texts = dataset[0].tolist()
string_label = dataset[1].tolist()

# 2.标签转数字（分类任务必须把字符串标签转成数字索引）
label_to_index = {label: i for i, label in enumerate(set(string_label))} # {'Weather-Query': 0}
numerical_labels = [label_to_index[label] for label in string_label] # 把每个文本对应的字符串标签转成数字 比如所有的Weather-Query都是0

# 字符词汇表构建
# 3.构建字符到索引的映射字典（字符级Vocab）
char_to_index = {'<pad>': 0} # 初始化：0留给填充字符<pad>
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index) # 词汇词表大小（所有唯一字符的数量）
max_len = 40 # 文本最大长度：超过40截断，不足40补0

# 生成字符级Bow向量
class CharBowDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors() # 生成Bow向量
    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)
        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size) # torch.Size([2823])
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        return self.bow_vectors[index], self.labels[index]

'''
将num_epochs增加到100
原始loss的变化：
Batch个数200,当前Batch Loss:0.11643272638320923
Batch个数250,当前Batch Loss:0.11520753800868988
Batch个数300,当前Batch Loss:0.06904946267604828
Batch个数350,当前Batch Loss:0.06047673895955086
Epoch[100/100],Loss:0.1098

增加了一个激活层和线形层后Loss的变化：
Batch个数150,当前Batch Loss:0.04528312757611275
Batch个数200,当前Batch Loss:0.024471400305628777
Batch个数250,当前Batch Loss:0.04409260302782059
Batch个数300,当前Batch Loss:0.02539755031466484
Batch个数350,当前Batch Loss:0.007087815087288618
Epoch[100/100],Loss:0.0382

把hidden_dim调整为20Loss的变化：
Batch个数150,当前Batch Loss:0.042288780212402344
Batch个数200,当前Batch Loss:0.053067807108163834
Batch个数250,当前Batch Loss:0.03215261548757553
Batch个数300,当前Batch Loss:0.02332152985036373
Batch个数350,当前Batch Loss:0.017548376694321632
Epoch[100/100],Loss:0.0361

'''

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        return out

char_dataset = CharBowDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据库-》batch数据 shuffle=True训练时打乱数据

hidden_dim = 20
hidden_dim2 = 64
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, hidden_dim2, output_dim)
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数 内自带softmax激活函数
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        optputs = model(inputs)
        loss = criterion(optputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch个数{idx},当前Batch Loss:{loss.item()}")
    print(f"Epoch[{epoch + 1}/{num_epochs}],Loss:{running_loss /len(dataloader):.4f}")

def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 1. 新文本处理:字符转数字+补0
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    # 2. 生成Bow向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1
    bow_vector = bow_vector.unsqueeze(0) # 增加维度（模型要求批量输入，即使单样本）

    # 3.模型预测（禁用梯度计算，提升速度）
    model.eval()
    with torch.no_grad():
        ouput = model(bow_vector) # 输出是每个类别的得分
    # 4.取得分最高的类别索引
    print(ouput, '查看输出的值是什么---===---')
    _, predicted_index = torch.max(ouput, 1)
    predicted_index = predicted_index.item()

    # 5.数字索引转回标签
    predicted_label = index_to_label[predicted_index]
    return predicted_label

index_to_label = {i:label for label, i in label_to_index.items()}
new_text = '帮我导航到北京'
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)

print(f"输入'{new_text}'预测为：{predicted_class}")

new_text_2 = '查询明天北京的天气'
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入'{new_text_2}'预测为：{predicted_class_2}")
