import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn as nn

import torch

# 1. 生成 X 轴数据
# 从 0 到 2*pi，生成 100 个点
x_numpy = np.linspace(-math.pi, math.pi, 100).reshape(-1, 1)
y_numpy = np.sin(x_numpy) + np.random.randn(100, 1)

# 转换为PyTorch张量
X = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成")
print("---" * 10)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, x):
        return self.network(x)


# 实例化模型
model = MLP()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 2000
loss_history = []

for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 打印最终学到的参数
print("\n训练完成！")

model.eval()

with torch.no_grad():
    y_predicate = model(X).numpy()

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(x_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(x_numpy, y_predicate, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
