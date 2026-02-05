"""
@Author  :  CAISIMIN
@Date    :  2026/1/27 16:49

功能：使用rnn、lstm、gru进行文本分类
实验数据：附在本文件最后
实验发现：
    1. loss下降速度：gru > lstm > rnn
    2. 预测效果：gru和lstm均能正确预测样本类别，rnn对两个测试样本的预测结果均错误
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 读取数据
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", names=["text", "label"])

# type : pandas.core.series.Series  -> list
texts = dataset["text"].tolist()
labels = dataset["label"].tolist()

# label编码
# label去重，创建词典
labels_to_index = {label: index for index, label in enumerate(set(labels))}
# 样本label映射为index
numberic_labels = [labels_to_index[label] for label in labels]

# text 词典构建
char_to_index = {'<pad>': 0}
for text in texts:
    for token in text:
        if token not in char_to_index:
            char_to_index[token] = len(char_to_index)

vocab_size = len(char_to_index)

# 设置最大长度
max_len = 40


# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, max_len, char_to_index):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_len = max_len
        self.char_to_index = char_to_index

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        根据单个样本的索引，获取样本的text和label
        Args:
            idx: 样本索引

        Returns:
            text
            label
        """
        text = self.texts[idx]
        indices = [self.char_to_index[token] for token in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# 创建lstm分类器
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 状态初始化机制
        # 自动初始化：LSTM会自动初始化隐藏状态h_0和细胞状态c_0为零张量
        # 形状规格：h_0和c_0的形状为(num_layers * num_directions, batch_size, hidden_dim)
        # 零状态：默认情况下，所有状态都初始化为0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # lstm_out: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim)
        # hidden_state/cell_state: (num_layers * num_directions, batch_size, hidden_dim)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        out = self.fc(hidden_state.squeeze(0))

        return out


# 创建rnn分类器
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim)
        rnn_out, hidden_state = self.rnn(embedded)

        out = self.fc(hidden_state.squeeze(0))
        return out


# 创建gru分类器
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


# 加载数据集
dataset = TextDataset(texts, numberic_labels, max_len, char_to_index)
dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(labels_to_index)


def train(model, data_loader, epochs_num, lr):
    """
    训练模型
    Args:
        model: 模型
        data_loader: 数据
        epochs_num: 训练轮次
        lr: 学习率

    Returns:

    """
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs_num):
        model.train()
        running_loss = 0.0
        for index, (inputs, labels) in enumerate(data_loader):
            # 清空梯度
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            running_loss += loss.item()
            if index % 50 == 0:
                print(f"Batch 个数 {index}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{epochs_num}], Loss: {running_loss / len(data_loader):.4f}")


def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(token, 0) for token in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {index:label for label, index in labels_to_index.items()}

# 测试样本
new_text = "帮我导航到北京"
new_text_2 = "查询明天北京的天气"

# rnn
rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
print("=============rnn模型训练=================")
train(rnn_model, dataLoader, 4, 0.001)
print("=============rnn模型预测=================")
rnn_predicted_class = classify_text(new_text, rnn_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' rnn预测为: '{rnn_predicted_class}'")
rnn_predicted_class_2 = classify_text(new_text_2, rnn_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' rnn预测为: '{rnn_predicted_class_2}'")

# lstm
lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
print("=============lstm模型训练=================")
train(lstm_model, dataLoader, 4, 0.001)
print("=============lstm模型预测=================")
lstm_predicted_class = classify_text(new_text, lstm_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' lstm预测为: '{lstm_predicted_class}'")
lstm_predicted_class_2 = classify_text(new_text_2, lstm_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' lstm预测为: '{lstm_predicted_class_2}'")

# gru
gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
print("=============gru模型训练=================")
train(gru_model, dataLoader, 4, 0.001)
print("=============gru模型预测=================")
gru_predicted_class = classify_text(new_text, gru_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' gru预测为: '{gru_predicted_class}'")
gru_predicted_class_2 = classify_text(new_text_2, gru_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' gru预测为: '{gru_predicted_class_2}'")




# =============rnn模型训练=================
# Batch 个数 0, 当前Batch Loss: 2.5099213123321533
# Batch 个数 50, 当前Batch Loss: 2.236379384994507
# Batch 个数 100, 当前Batch Loss: 2.5285239219665527
# Batch 个数 150, 当前Batch Loss: 2.2902302742004395
# Batch 个数 200, 当前Batch Loss: 2.3367199897766113
# Batch 个数 250, 当前Batch Loss: 2.3417866230010986
# Batch 个数 300, 当前Batch Loss: 2.4428253173828125
# Batch 个数 350, 当前Batch Loss: 2.3643715381622314
# Batch 个数 0, 当前Batch Loss: 2.369286060333252
# Batch 个数 50, 当前Batch Loss: 2.2764086723327637
# Batch 个数 100, 当前Batch Loss: 2.294663667678833
# Batch 个数 150, 当前Batch Loss: 2.3128795623779297
# Batch 个数 200, 当前Batch Loss: 2.3831353187561035
# Batch 个数 250, 当前Batch Loss: 2.341752529144287
# Batch 个数 300, 当前Batch Loss: 2.257319927215576
# Batch 个数 350, 当前Batch Loss: 2.425100564956665
# Batch 个数 0, 当前Batch Loss: 2.3454699516296387
# Batch 个数 50, 当前Batch Loss: 2.2764389514923096
# Batch 个数 100, 当前Batch Loss: 2.3101091384887695
# Batch 个数 150, 当前Batch Loss: 2.3016550540924072
# Batch 个数 200, 当前Batch Loss: 2.288196325302124
# Batch 个数 250, 当前Batch Loss: 2.3139352798461914
# Batch 个数 300, 当前Batch Loss: 2.4135098457336426
# Batch 个数 350, 当前Batch Loss: 2.303119421005249
# Batch 个数 0, 当前Batch Loss: 2.567460060119629
# Batch 个数 50, 当前Batch Loss: 2.2422690391540527
# Batch 个数 100, 当前Batch Loss: 2.3019328117370605
# Batch 个数 150, 当前Batch Loss: 2.3629701137542725
# Batch 个数 200, 当前Batch Loss: 2.2214505672454834
# Batch 个数 250, 当前Batch Loss: 2.3682844638824463
# Batch 个数 300, 当前Batch Loss: 2.236145496368408
# Batch 个数 350, 当前Batch Loss: 2.257791042327881
# Epoch [4/4], Loss: 2.3586
# =============rnn模型预测=================
# 输入 '帮我导航到北京' rnn预测为: 'Travel-Query'
# 输入 '查询明天北京的天气' rnn预测为: 'Travel-Query'
# =============lstm模型训练=================
# Batch 个数 0, 当前Batch Loss: 2.443918228149414
# Batch 个数 50, 当前Batch Loss: 2.3040342330932617
# Batch 个数 100, 当前Batch Loss: 2.3249151706695557
# Batch 个数 150, 当前Batch Loss: 2.337625026702881
# Batch 个数 200, 当前Batch Loss: 2.374392032623291
# Batch 个数 250, 当前Batch Loss: 2.2618610858917236
# Batch 个数 300, 当前Batch Loss: 2.424351453781128
# Batch 个数 350, 当前Batch Loss: 2.504298210144043
# Batch 个数 0, 当前Batch Loss: 2.2853167057037354
# Batch 个数 50, 当前Batch Loss: 2.3088619709014893
# Batch 个数 100, 当前Batch Loss: 2.2424535751342773
# Batch 个数 150, 当前Batch Loss: 2.3207225799560547
# Batch 个数 200, 当前Batch Loss: 2.3681797981262207
# Batch 个数 250, 当前Batch Loss: 2.2936699390411377
# Batch 个数 300, 当前Batch Loss: 2.043604850769043
# Batch 个数 350, 当前Batch Loss: 1.3648269176483154
# Batch 个数 0, 当前Batch Loss: 1.7692790031433105
# Batch 个数 50, 当前Batch Loss: 1.3417611122131348
# Batch 个数 100, 当前Batch Loss: 1.8290953636169434
# Batch 个数 150, 当前Batch Loss: 1.3249595165252686
# Batch 个数 200, 当前Batch Loss: 1.0071450471878052
# Batch 个数 250, 当前Batch Loss: 1.3373219966888428
# Batch 个数 300, 当前Batch Loss: 1.4172815084457397
# Batch 个数 350, 当前Batch Loss: 0.8253514766693115
# Batch 个数 0, 当前Batch Loss: 0.8347912430763245
# Batch 个数 50, 当前Batch Loss: 1.121488094329834
# Batch 个数 100, 当前Batch Loss: 0.8268621563911438
# Batch 个数 150, 当前Batch Loss: 0.6784203052520752
# Batch 个数 200, 当前Batch Loss: 0.7320283055305481
# Batch 个数 250, 当前Batch Loss: 0.8104021549224854
# Batch 个数 300, 当前Batch Loss: 0.5432518720626831
# Batch 个数 350, 当前Batch Loss: 0.588982343673706
# Epoch [4/4], Loss: 0.8385
# =============lstm模型预测=================
# 输入 '帮我导航到北京' lstm预测为: 'Travel-Query'
# 输入 '查询明天北京的天气' lstm预测为: 'Weather-Query'
# =============gru模型训练=================
# Batch 个数 0, 当前Batch Loss: 2.470816135406494
# Batch 个数 50, 当前Batch Loss: 2.254007577896118
# Batch 个数 100, 当前Batch Loss: 2.221451997756958
# Batch 个数 150, 当前Batch Loss: 1.7889986038208008
# Batch 个数 200, 当前Batch Loss: 1.2610931396484375
# Batch 个数 250, 当前Batch Loss: 0.8101521134376526
# Batch 个数 300, 当前Batch Loss: 0.7222349643707275
# Batch 个数 350, 当前Batch Loss: 0.8720210194587708
# Batch 个数 0, 当前Batch Loss: 0.7992048263549805
# Batch 个数 50, 当前Batch Loss: 0.5573256611824036
# Batch 个数 100, 当前Batch Loss: 0.4241059422492981
# Batch 个数 150, 当前Batch Loss: 0.6791122555732727
# Batch 个数 200, 当前Batch Loss: 0.5865452885627747
# Batch 个数 250, 当前Batch Loss: 0.8066839575767517
# Batch 个数 300, 当前Batch Loss: 0.5779067277908325
# Batch 个数 350, 当前Batch Loss: 0.45559853315353394
# Batch 个数 0, 当前Batch Loss: 0.3400544822216034
# Batch 个数 50, 当前Batch Loss: 0.3942610025405884
# Batch 个数 100, 当前Batch Loss: 0.3568111062049866
# Batch 个数 150, 当前Batch Loss: 0.2951899766921997
# Batch 个数 200, 当前Batch Loss: 0.1856519877910614
# Batch 个数 250, 当前Batch Loss: 0.14262448251247406
# Batch 个数 300, 当前Batch Loss: 0.06632570922374725
# Batch 个数 350, 当前Batch Loss: 0.14795264601707458
# Batch 个数 0, 当前Batch Loss: 0.3664044439792633
# Batch 个数 50, 当前Batch Loss: 0.07872048020362854
# Batch 个数 100, 当前Batch Loss: 0.1742788702249527
# Batch 个数 150, 当前Batch Loss: 0.07130379229784012
# Batch 个数 200, 当前Batch Loss: 0.10668862611055374
# Batch 个数 250, 当前Batch Loss: 0.24650245904922485
# Batch 个数 300, 当前Batch Loss: 0.2778960168361664
# Batch 个数 350, 当前Batch Loss: 0.14011168479919434
# Epoch [4/4], Loss: 0.2293
# =============gru模型预测=================
# 输入 '帮我导航到北京' gru预测为: 'Travel-Query'
# 输入 '查询明天北京的天气' gru预测为: 'Weather-Query'
