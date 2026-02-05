import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 2 * np.pi
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy) + np.random.randn(100, 1)
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


# 2. 定义多层神经网络模型
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        # 构建多层全连接网络
        self.layers = nn.Sequential(
            nn.Linear(1, 32),  # 输入层：1个特征 -> 32个隐藏单元
            nn.ReLU(),  # 激活函数，引入非线性
            nn.Linear(32, 64),  # 隐藏层1：32 -> 64
            nn.ReLU(),
            nn.Linear(64, 32),  # 隐藏层2：64 -> 32
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出层：32 -> 1（预测sin值）
        )

    def forward(self, x):
        # 前向传播
        return self.layers(x)

model = SinNet()

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
# a * x + b 《 - 》  y'

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")


# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值、
X_test = torch.linspace(0, 2 * np.pi, 500).reshape(-1, 1).float()
with torch.no_grad():
    y_predicted = model(X_test)

# 转换为numpy数组用于绘图
X_test_numpy = X_test.numpy()
y_pred_numpy = y_predicted.numpy()
y_true_numpy = np.sin(X_test_numpy)  # 真实的sin值

plt.subplot(2, 1, 2)
plt.scatter(X_numpy, y_numpy, label='Raw Data (with noise)', color='blue', alpha=0.3, s=10)
plt.plot(X_test_numpy, y_true_numpy, label='True sin(x)', color='red', linewidth=2)
plt.plot(X_test_numpy, y_pred_numpy, label='Fitted Curve', color='orange', linewidth=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Fitting sin(x) with Multi-layer Neural Network')
plt.legend()
plt.grid(True)
plt.show()
