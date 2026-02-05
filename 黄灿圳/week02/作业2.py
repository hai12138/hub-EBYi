import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 生成模拟数据
# X_numpy = np.random.rand(100, 1) * 10
# y_numpy = np.sin(X_numpy)
X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1) # 批量训练， 100*1
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) / 20

# 将NumPy数组转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

class CharBoWDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx][0]


class SimpleClassifier(nn.Module):

    def __init__(self, layer_num, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        super(SimpleClassifier, self).__init__()
        layers = []

        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 动态添加隐藏层
        # layer_num 表示隐藏层的数量
        for _ in range(layer_num - 1):  # 减去输入层
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        # 封装为 Sequential
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)


char_dataset = CharBoWDataset(X, y) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=64, shuffle=True) # 读取批量数据集 -》 batch数据

layer_nums = [3]
hidden_dims = [128]
models = []
for layer_num in layer_nums:
    for hidden_dim in hidden_dims:
        model = SimpleClassifier(layer_num, 1,  hidden_dim, 1) # 维度和精度有什么关系？
        loss_fn = nn.MSELoss() # 损失函数 内部自带激活函数，softmax
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # epoch： 将数据集整体迭代训练一次
        # batch： 数据集汇总为一批训练一次
        num_epochs = 5000
        finalLoss = 1
        for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
            model.train()
            running_loss = 0.0
            for idx, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if idx % 50 == 0:
                    print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
            finalLoss = running_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {finalLoss:.4f}")
        models.append({"ln":layer_num,"hd":hidden_dim,"loss":finalLoss,"model": model})

print("==="*10)
models.sort(key=lambda x: x["loss"])
for model in models:
    print(f"层数{model.get("ln")} 节点数{model.get("hd")} Loss: {model.get('loss')}")

model = models[0].get("model")


# 将模型切换到评估模式，这在训练结束后是好习惯
model.eval() # 主动关闭dropout

# 禁用梯度计算，因为我们不再训练
with torch.no_grad():
    y_predicted = model(X).numpy() # 使用训练好的模型进行预测

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
