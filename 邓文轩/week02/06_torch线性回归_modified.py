import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np  # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import random



# 1. 生成模拟数据 (与之前相同)
# 形状为 (1000, 1) 的二维数组，其中包含 1000 个在 [0, 1) 范围内均匀分布的随机浮点数。
X_numpy = np.random.rand(1000, 1) * 30

# y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1) # randn是均值为0，标准差为1的高斯分布
y_numpy = np.sin(X_numpy) + np.random.randn(1000, 1)

X = torch.from_numpy(X_numpy).float()  # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 直接创建参数张量 a 和 b，创建多个线形层拟合sin
# torch.randn() 生成随机值作为初始值。
# y = a * x + b
# requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
a1 = torch.randn((1, 64), dtype=torch.float) * np.sqrt(2 / 1)
b1 = torch.randn(64, requires_grad=True, dtype=torch.float)

a2 = torch.randn((64, 128), dtype=torch.float) * np.sqrt(2 / 64)
b2 = torch.randn(128, requires_grad=True, dtype=torch.float)

a3 = torch.randn((128, 1), dtype=torch.float) * np.sqrt(2 / 128)
b3 = torch.randn(1, requires_grad=True, dtype=torch.float)

a1.requires_grad_()
a2.requires_grad_()
a3.requires_grad_()

print(f"初始参数 a1: {a1.shape}")
print(f"初始参数 b1: {b1.shape}")
print(f"初始参数 a2: {a2.shape}")
print(f"初始参数 b2: {b2.shape}")
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss()  # 回归任务
# a * x + b 《 - 》  y'

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.Adam([a1, b1, a2, b2, a3, b3], lr=0.01)  # 优化器，基于 a b 梯度 自动更新


def forward(x):
    x = torch.matmul(x, a1) + b1 # (batch_size,1) -> (batch_size,64)
    x = torch.tanh(x)
    x = torch.matmul(x, a2) + b2 # (batch_size,64) -> (batch_size,128)
    x = torch.tanh(x)
    x = torch.matmul(x, a3) + b3 # (batch_size,128) -> (batch_size,1)
    return x


# 4. 训练模型
num_epochs = 2000
batch_size = 1000

# 样本数量n
n = X.shape[0]
print("样本数量n:",n)
batch_num = n // batch_size
print("batch数量：",batch_num)
for epoch in range(num_epochs):
    epoch_loss = []
    random_index = list(range(n))
    random.shuffle(random_index)
    for i in range(0, n, batch_size):
        batch_x = X[random_index[i:i+batch_size]]
        batch_y = y[random_index[i:i+batch_size]]
        # 前向传播：手动计算 y_pred = a * X + b
        y_pred = forward(batch_x)
        # 计算损失
        loss = loss_fn(y_pred, batch_y)
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        epoch_loss.append(loss.item())

    # 每10个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {sum(epoch_loss)/batch_size:.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
# a_learned = a.item()
# b_learned = b.item()
print(f"拟合的斜率 a1: {a1}")
print(f"拟合的截距 b1: {b1}")

print(f"拟合的斜率 a2: {a2}")
print(f"拟合的截距 b2: {b2}")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = forward(X)
    y_predicted = y_predicted.numpy()

    print(y_predicted.shape)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.scatter(X_numpy, y_predicted, label=f'Model', color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig("作业2_拟合sin曲线.png", dpi=300, bbox_inches="tight")
plt.show()
