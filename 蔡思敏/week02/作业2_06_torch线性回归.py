"""
    改动点：
    1. 构建sin函数
    2. 构建一个4层的网络
    3. 优化器使用Adam

    运行结果 附在该文件最后
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据
X_numpy = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)

y_numpy = np.sin(X_numpy)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 创建多层网络
class MultiLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(MultiLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


input_dim = X.shape[1]
hidden_dim1 = 64
hidden_dim2 = 64
hidden_dim3 = 32
output_dim = 1

model = MultiLayerNet(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model: y = sin(x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('作业2可视化.png', dpi=300, bbox_inches='tight')
plt.show()

# 数据生成完成。
# ------------------------------
# Epoch [100/1000], Loss: 0.0053
# Epoch [200/1000], Loss: 0.0002
# Epoch [300/1000], Loss: 0.0002
# Epoch [400/1000], Loss: 0.0002
# Epoch [500/1000], Loss: 0.0010
# Epoch [600/1000], Loss: 0.0003
# Epoch [700/1000], Loss: 0.0004
# Epoch [800/1000], Loss: 0.0004
# Epoch [900/1000], Loss: 0.0002
# Epoch [1000/1000], Loss: 0.0002
#
# 训练完成！
# ------------------------------
#
# Process finished with exit code 0

