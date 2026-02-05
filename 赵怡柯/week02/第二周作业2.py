# 2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成模拟数据
X_numpy = np.linspace(0, 10, 1000).reshape(-1, 1) # 0-10之间的1000个点，均匀分布
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape) # *X_numpy.shape：确保噪声和

X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

# 因为数据已经是tensor了，所以可以直接使用TensorDataset，不需要重新自定义类
dataset = TensorDataset(X, y)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络类
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        # 构建多层全连接网络（带非线性激活）
        self.layers = nn.Sequential(
            nn.Linear(1, 32),  # 输入层：1个特征 → 32个隐藏单元
            nn.ReLU(),  # 激活函数，引入非线性（关键！）
            nn.Linear(32, 64),  # 隐藏层1
            nn.ReLU(),
            nn.Linear(64, 32),  # 隐藏层2
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出层：回归任务输出1个值
        )

    def forward(self, x):
        # 前向传播逻辑
        return self.layers(x)


# 实例化模型
model = SinNet()
print("神经网络模型结构：")
print(model)
print("---" * 10)



# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务

# 使用Adam优化器（比SGD更适合非线性拟合），传入模型所有参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam优化器，更适合非线性拟合


# 4. 训练模型
num_epochs = 1000
loss_history = [] # 记录每轮损失
for epoch in range(num_epochs):
    epoch_loss = 0.0
    batch_count = 0

    for batch_X, batch_y in dataloader:
        # 前向传播
        y_pred = model(X)

        # 计算损失
        loss = loss_fn(y_pred, y)

        # 保存当前损失值
        loss_history.append(loss.item())

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count
    loss_history.append(avg_loss)

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 5. 绘制结果
# 用训练好的模型预测（禁用梯度计算）
with torch.no_grad():
    y_predicted = model(X).numpy()  # 转numpy方便绘图

# 绘制双图：损失变化 + 拟合结果
plt.figure(figsize=(12, 8))

# 子图1：损失变化曲线
plt.subplot(2, 1, 1)
plt.plot(loss_history, color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)

# 子图2：sin函数拟合结果
plt.subplot(2, 1, 2)
# 原始带噪声数据
plt.scatter(X_numpy, y_numpy, label='Raw Data (sin(x)+noise)', color='blue', alpha=0.3, s=5)
# 网络拟合曲线
plt.plot(X_numpy, y_predicted, label='Fitted Curve', color='red', linewidth=2)
# 纯sin(x)曲线（参考）
plt.plot(X_numpy, np.sin(X_numpy), label='Pure sin(x)', color='orange', linestyle='--', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('sin(x) Fitting with Multi-Layer Network')

plt.tight_layout()  # 自动调整子图间距
plt.show()
