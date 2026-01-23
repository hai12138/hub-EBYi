import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据
X_numpy = np.linspace(0, 10, 100)  # 在 [0, 10] 范围内生成 100 个点
y_numpy = np.sin(X_numpy)  # 目标是 sin(x)

# 将数据转换为 PyTorch 的 tensor
X = torch.from_numpy(X_numpy).float().view(-1, 1)  # 改变形状为 (100, 1)
y = torch.from_numpy(y_numpy).float().view(-1, 1)

print("数据生成完成。")
print("---" * 10)

# 2. 构建一个多层感知器（MLP）
# 输入层有1个神经元，输出层有1个神经元。
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),    # 输入层到隐藏层
    torch.nn.ReLU(),           # 激活函数
    torch.nn.Linear(64, 64),   # 隐藏层到隐藏层
    torch.nn.ReLU(),           # 激活函数
    torch.nn.Linear(64, 1)     # 隐藏层到输出层
)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 使用 SGD 优化器

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
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

# 5. 打印训练完成的模型参数
print("\n训练完成！")

# 6. 绘制结果
# 使用模型进行预测
with torch.no_grad():
    y_predicted = model(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='True data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted.numpy(), label='Fitted model', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
