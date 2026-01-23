import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 2 * np.pi
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 直接创建参数张量 a 和 b
# torch.randn() 生成随机值作为初始值。
# y = a * x + b
# requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
# a * x + b 《 - 》  y'

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.SGD([a, b], lr=0.1) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    y_pred = torch.sin(a * X + b)


    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 10}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
a_learned = a.item()
b_learned = b.item()
print(f"拟合的 a: {a_learned:.4f}")
print(f"拟合的 b: {b_learned:.4f}")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = torch.sin(a_learned * X + b_learned).detach().numpy()

# # 核心修改：对X和y_predicted按X升序排序
# # 1. 提取X的一维数组（去掉多余维度），获取排序索引
# sorted_indices = np.argsort(X_numpy, axis=0)
# # 2. 按索引重新排列X和y_predicted
# X_sorted = X_numpy[sorted_indices].reshape(-1)  # 展平为一维
# y_pred_sorted = y_predicted[sorted_indices].reshape(-1)

plt.figure(figsize=(10, 6))
# plt.scatter(X_sorted, y_pred_sorted, label='Raw data', color='blue', alpha=0.6)
# plt.plot(X_sorted, y_pred_sorted, label=f'Model: y = sin({a_learned:.2f}x + {b_learned:.2f})', color='red', linewidth=2)
# 图形错乱是因为plt按照点随机生成的顺序连接,但是如果进行排序又太规范了.


plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model: y = sin({a_learned:.2f}x + {b_learned:.2f})', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
