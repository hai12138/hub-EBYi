import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# 收集数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 中文分词
def chinese_cut(text):
    words = jieba.lcut(str(text).strip())
    return " ".join([w for w in words if w])

dataset["cut_text"] = dataset[0].apply(chinese_cut)

# TF-IDF特征提取
vectorizer = TfidfVectorizer(max_features=2000, norm='l2')
X = vectorizer.fit_transform(dataset["cut_text"]).toarray()

# 标签编码
label2id = {label: idx for idx, label in enumerate(dataset[1].unique())}
y = np.array([label2id[label] for label in dataset[1]])
num_classes = len(label2id)
input_dim = X.shape[1]  # 输入维度=TF-IDF特征数（2000）

# 划分数据集并转Tensor
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 转PyTorch Tensor（MLP需要浮点型输入）
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 建模
class TextMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(TextMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 输入层→隐藏层1
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2) # 隐藏层1→隐藏层2
        self.fc3 = nn.Linear(hidden_dim2, num_classes) # 隐藏层2→输出层（类别数）
        # 激活函数（解决线性不可分问题）
        self.relu = nn.ReLU()
        # Dropout（防止过拟合）
        self.dropout = nn.Dropout(0.2)
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 前向传播：输入→隐藏层1→ReLU→Dropout→隐藏层2→ReLU→Dropout→输出层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        y_pred = self.fc3(x)  # 输出：[batch_size, num_classes]
        
        # 计算损失（训练时用）
        if y is not None:
            return self.loss_fn(y_pred, y)
        else:
            return y_pred

# 模型初始化
hidden_dim1 = 512  # 第一层隐藏层维度
hidden_dim2 = 256  # 第二层隐藏层维度
model = TextMLP(input_dim, hidden_dim1, hidden_dim2, num_classes)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
epochs = 15
batch_size = 32
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    # 按批次训练
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # 前向传播计算损失
        loss = model(batch_x, batch_y)
        
        # 反向传播更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # 打印每轮损失
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train):.4f}")

# 模型评估
model.eval()
with torch.no_grad():
    y_pred = model(X_test)  # 预测值：[test_size, num_classes]
    y_pred_id = torch.argmax(y_pred, dim=1)  # 取概率最大的类别
    # 转numpy计算评估指标
    y_test_np = y_test.numpy()
    y_pred_np = y_pred_id.numpy()
    
    # 输出准确率和详细报告
    acc = accuracy_score(y_test_np, y_pred_np)
    print(f"\n测试集准确率：{acc:.2f}")

# 预测
id2label = {idx: label for label, idx in label2id.items()}
print(label2id, '查看label2id的值') # {'Travel-Query': 0, 'Music-Play': 1, 'FilmTele-Play': 2 .....}
while True:
    user_input = input("请输入句子：").strip()
    if user_input == "退出":
        break
    if not user_input:
        print("输入不能为空！")
        continue
    
    # 预处理（和训练数据一致）
    cut_input = chinese_cut(user_input)
    if not cut_input:
        print("输入无有效语义！")
        continue
    
    # 转TF-IDF特征
    new_X = vectorizer.transform([cut_input]).toarray()
    new_X = torch.tensor(new_X, dtype=torch.float32)
    
    # 预测
    model.eval()
    with torch.no_grad():
        pred = model(new_X)
        pred_id = torch.argmax(pred, dim=1).item()
        pred_label = id2label[pred_id]
    
    print(f"分类结果：{pred_label}\n")