import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(500, 1) * 10

# y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)
y_numpy = 3 * np.sin(2 * X_numpy) + 1 + np.random.randn(500, 1)* 0.2
X_deal = X_numpy / 10

X = torch.from_numpy(X_deal).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 定义模型
class regression(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out


hidden_dim1 = 32
hidden_dim2 = 32
model = regression(1, hidden_dim1, hidden_dim2, 1)
print(model)
print("---" * 10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 3000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # 前向传播：手动计算 y_pred = a * X + b
    # y_pred = a * X + b
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
plt.figure(figsize=(10, 6))

# 画蓝点：原始数据乱序没关系，因为是散点图 (scatter)
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.5)

# 画红线：必须用“有序”的 X 来画，否则会乱连线
model.eval()
with torch.no_grad():
    # 1. 生成一个从小到大排列整齐的 X 轴 (0到10之间生成200个点)
    X_plot = torch.linspace(0, 10, 200).unsqueeze(1)
    y_plot = model(X_plot / 10)

# 3. 此时 X_plot 和 y_plot 都是有序的，画出来就是丝滑曲线
plt.plot(X_plot.numpy(), y_plot.numpy(), label=f'Model',
         color='red', linewidth=3)

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
