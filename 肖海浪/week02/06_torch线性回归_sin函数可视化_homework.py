import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# =========================
# 1) 构建 sin(x) 数据
# =========================

N = 200  # 样本数量（越多曲线越密，但训练更慢）

# 生成 [0, 2π] 上的等间隔 x
# reshape(-1, 1) 是关键：让 X 形状变成 [N, 1]，符合 Linear 输入
X_numpy = np.linspace(0, 2 * np.pi, N).reshape(-1, 1)

# 真值曲线（不加噪声的 sin）
y_true_numpy = np.sin(X_numpy)

# 可选：加一点噪声，让任务更像真实回归
noise = 0.05 * np.random.randn(N, 1)
y_numpy = y_true_numpy + noise

# 转成 torch.Tensor（用 float32）
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin 数据生成完成。  X shapes:", X.shape, " y shapes:", y.shape)

# =========================
# 2) 定义多层网络（MLP）来拟合 sin(x)
# =========================
class MLP(torch.nn.Module):
    """
    一个简单的多层感知机：
    输入 1 维（x）
    输出 1 维（y）
    中间用多层 Linear + ReLU 提供非线性表达能力
    """
    def __init__(self):
        super().__init__()

        # nn.Sequential：按顺序把层串起来，forward 时自动依次执行
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 64),   # 1 -> 64：把输入映射到更高维特征空间
            torch.nn.ReLU(),          # 非线性：没有它，多层线性叠加仍是线性，拟合不了 sin
            torch.nn.Linear(64, 64),  # 64 -> 64：增强表达能力
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)    # 64 -> 1：回归输出
        )

    def forward(self, x):
        # x 形状：[batch, 1]
        # 输出形状：[batch, 1]
        return self.net(x)

model = MLP()


# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
# a * x + b 《 - 》  y'

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 1600
loss_history = []

for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    y_pred = model(X)

    # 计算损失loss
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    loss_history.append(loss.item())

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")

# =========================
# 5) 可视化：真值曲线 vs 拟合曲线
# =========================
model.eval()  # 切换到评估模式（本例没 dropout/bn 也建议写规范）
with torch.no_grad():  # 推理时不记录梯度，省内存更快
    y_fit = model(X).cpu().numpy()

# 画曲线时要排序，否则线会乱（非常常见的坑）
sort_idx = np.argsort(X_numpy[:, 0])
X_sorted = X_numpy[sort_idx]
y_true_sorted = y_true_numpy[sort_idx]
y_fit_sorted = y_fit[sort_idx]

plt.figure(figsize=(10, 5))
plt.scatter(X_numpy, y_numpy, s=15, alpha=0.6, label="Samples (sin + noise)")
plt.plot(X_sorted, y_true_sorted, linewidth=2, label="True sin(x)")
plt.plot(X_sorted, y_fit_sorted, linewidth=2, label="MLP fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("MLP Fitting sin(x)")
plt.grid(True)
plt.legend()
plt.show()

# =========================
# 6) 可视化：Loss 曲线（观察是否收敛）
# =========================
plt.figure(figsize=(10, 4))
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

