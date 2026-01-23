import torch
import numpy as np
import matplotlib.pyplot as plt

# 配置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成sin函数模拟数据
X_numpy = np.linspace(0, 4 * np.pi, 1000).reshape(-1, 1)  # 生成0到4π之间的数据点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数加上少量噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),  # 输入层到第一个隐藏层，64个神经元
    torch.nn.ReLU(),  # 激活函数
    torch.nn.Linear(64, 32),  # 第一个隐藏层到第二个隐藏层，32个神经元
    torch.nn.ReLU(),  # 激活函数
    torch.nn.Linear(32, 16),  # 第二个隐藏层到第三个隐藏层，16个神经元
    torch.nn.ReLU(),  # 激活函数
    torch.nn.Linear(16, 1)  # 输出层
)

print("多层神经网络结构:")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam优化器，通常比SGD更高效

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print(f"最终损失: {loss.item():.6f}")
print("---" * 10)

# 5. 使用训练好的模型预测
model.eval()  # 设置为评估模式
with torch.no_grad():  # 不计算梯度
    y_predicted = model(X)

# 6. 可视化结果
plt.figure(figsize=(12, 8))
plt.scatter(X_numpy, y_numpy, label='原始数据 (含噪声)', color='blue', alpha=0.3, s=10)
plt.plot(X_numpy, np.sin(X_numpy), label='真实sin函数', color='green', linewidth=2, linestyle='--')
plt.plot(X_numpy, y_predicted.numpy(), label='神经网络拟合结果', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('多层神经网络拟合sin函数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
